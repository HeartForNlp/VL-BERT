import os
import pprint
import shutil
import inspect
import random
import time
import pickle
from collections import namedtuple

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from common.utils.create_logger import create_logger
from common.utils.misc import summary_parameters, bn_fp16_half_eval
from common.utils.load import smart_resume, smart_partial_load_model_state_dict
from common.trainer import to_cuda
from common.metrics.composite_eval_metric import CompositeEvalMetric
from common.metrics import pretrain_metrics
from common.callbacks.batch_end_callbacks.speedometer import Speedometer
from common.callbacks.epoch_end_callbacks.validation_monitor import ValidationMonitor
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.lr_scheduler import WarmupMultiStepLR
from common.nlp.bert.optimization import AdamW, WarmupLinearSchedule
from common.utils.multi_task_dataloader import MultiTaskDataLoader
from pretrain.data.build import make_dataloader, make_dataloaders
from pretrain.modules import *
from pretrain.function.val import do_validation

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")


def train_net(args, config):
    # setup logger
    logger, final_output_path = create_logger(config.OUTPUT_PATH,
                                              args.cfg,
                                              config.DATASET[0].TRAIN_IMAGE_SET if isinstance(config.DATASET, list)
                                              else config.DATASET.TRAIN_IMAGE_SET,
                                              split='train')
    model_prefix = os.path.join(final_output_path, config.MODEL_PREFIX)
    if args.log_dir is None:
        args.log_dir = os.path.join(final_output_path, 'tensorboard_logs')

    pprint.pprint(args)
    logger.info('training args:{}\n'.format(args))
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # manually set random seed
    if config.RNG_SEED > -1:
        random.seed(config.RNG_SEED)
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = False
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    if args.dist:
        model = eval(config.MODULE)(config)
        local_rank = int(os.environ.get('LOCAL_RANK') or 0)
        config.GPUS = str(local_rank)
        torch.cuda.set_device(local_rank)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'] or 23456)
        world_size = int(os.environ['WORLD_SIZE'] or 1)
        rank = int(os.environ['RANK'] or 0)
        if args.slurm:
            distributed.init_process_group(backend='nccl')
        else:
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank,
                group_name='mtorch')
        print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        config.GPUS = str(local_rank)
        model = model.cuda()
        if not config.TRAIN.FP16:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        if rank == 0:
            summary_parameters(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
                               logger)
            shutil.copy(args.cfg, final_output_path)
            shutil.copy(inspect.getfile(eval(config.MODULE)), final_output_path)

        writer = None
        if args.log_dir is not None:
            tb_log_dir = os.path.join(args.log_dir, 'rank{}'.format(rank))
            if not os.path.exists(tb_log_dir):
                os.makedirs(tb_log_dir)
            writer = SummaryWriter(log_dir=tb_log_dir)

        if isinstance(config.DATASET, list):
            train_loaders_and_samplers = make_dataloaders(config,
                                                          mode='train',
                                                          distributed=True,
                                                          num_replicas=world_size,
                                                          rank=rank,
                                                          expose_sampler=True)

            train_loader = MultiTaskDataLoader([loader for loader, _ in train_loaders_and_samplers])
            train_sampler = train_loaders_and_samplers[0][1]
        else:
            train_loader, train_sampler = make_dataloader(config,
                                                          mode='train',
                                                          distributed=True,
                                                          num_replicas=world_size,
                                                          rank=rank,
                                                          expose_sampler=True)

        batch_size = world_size * (sum(config.TRAIN.BATCH_IMAGES)
                                   if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                   else config.TRAIN.BATCH_IMAGES)
        if config.TRAIN.GRAD_ACCUMULATE_STEPS > 1:
            batch_size = batch_size * config.TRAIN.GRAD_ACCUMULATE_STEPS
        base_lr = config.TRAIN.LR * batch_size
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if _k in n],
                                         'lr': base_lr * _lr_mult}
                                        for _k, _lr_mult in config.TRAIN.LR_MULT]
        optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters()
                                                        if all([_k not in n for _k, _ in config.TRAIN.LR_MULT])]})
        if config.TRAIN.OPTIMIZER == 'SGD':
            optimizer = optim.SGD(optimizer_grouped_parameters,
                                  lr=config.TRAIN.LR * batch_size,
                                  momentum=config.TRAIN.MOMENTUM,
                                  weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(optimizer_grouped_parameters,
                                   lr=config.TRAIN.LR * batch_size,
                                   weight_decay=config.TRAIN.WD)
        elif config.TRAIN.OPTIMIZER == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=config.TRAIN.LR * batch_size,
                              betas=(0.9, 0.999),
                              eps=1e-6,
                              weight_decay=config.TRAIN.WD,
                              correct_bias=True)
        else:
            raise ValueError('Not support optimizer {}!'.format(config.TRAIN.OPTIMIZER))
        total_gpus = world_size

    else:
        #os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
        model = eval(config.MODULE)(config)
        summary_parameters(model, logger)
        shutil.copy(args.cfg, final_output_path)
        shutil.copy(inspect.getfile(eval(config.MODULE)), final_output_path)
        num_gpus = len(config.GPUS.split(','))
        assert num_gpus <= 1 or (not config.TRAIN.FP16), "Not support fp16 with torch.nn.DataParallel. " \
                                                         "Please use amp.parallel.DistributedDataParallel instead."
        total_gpus = num_gpus
        rank = None
        writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir is not None else None

        # model
        print("Start parallel model")
        if num_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=[int(d) for d in config.GPUS.split(',')]).cuda()
        else:
            torch.cuda.set_device(int(config.GPUS))
            model.cuda()

        # loader
        print("Start loading data")
        if isinstance(config.DATASET, list):
            train_loaders = make_dataloaders(config, mode='train', distributed=False)
            train_loader = MultiTaskDataLoader(train_loaders)
        else:
            train_loader = make_dataloader(config, mode='train', distributed=False)
        train_sampler = None

        batch_size = num_gpus * (sum(config.TRAIN.BATCH_IMAGES) if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                 else config.TRAIN.BATCH_IMAGES)

    # partial load pretrain state dict
    if config.NETWORK.PARTIAL_PRETRAIN != "":
        pretrain_state_dict = torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=lambda storage, loc: storage)['state_dict']
        prefix_change = [prefix_change.split('->') for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]
        if len(prefix_change) > 0:
            pretrain_state_dict_parsed = {}
            for k, v in pretrain_state_dict.items():
                no_match = True
                for pretrain_prefix, new_prefix in prefix_change:
                    if k.startswith(pretrain_prefix):
                        k = new_prefix + k[len(pretrain_prefix):]
                        pretrain_state_dict_parsed[k] = v
                        no_match = False
                        break
                if no_match:
                    pretrain_state_dict_parsed[k] = v
            pretrain_state_dict = pretrain_state_dict_parsed
        smart_partial_load_model_state_dict(model, pretrain_state_dict)


    # batch end callbacks
    batch_size = len(config.GPUS.split(',')) * (sum(config.TRAIN.BATCH_IMAGES)
                                                if isinstance(config.TRAIN.BATCH_IMAGES, list)
                                                else config.TRAIN.BATCH_IMAGES)
    batch_end_callbacks = [Speedometer(batch_size, config.LOG_FREQUENT,
                                       batches_per_epoch=len(train_loader),
                                       epochs=1)]

    # broadcast parameter from rank 0 before training start
    if args.dist:
        for v in model.state_dict().values():
            distributed.broadcast(v, src=0)

    # set net to train mode
    model.eval()


    # init end time
    end_time = time.time()
    print("##################################################################################################")
    print("ca va commencer")

    # Parameter to pass to batch_end_callback
    BatchEndParam = namedtuple('BatchEndParams',
                               ['epoch',
                                'nbatch',
                                'rank',
                                'add_step',
                                'data_in_time',
                                'data_transfer_time',
                                'forward_time',
                                'backward_time',
                                'optimizer_time',
                                'metric_time',
                                'eval_metric',
                                'locals'])

    def _multiple_callbacks(callbacks, *args, **kwargs):
        """Sends args and kwargs to any configured callbacks.
        This handles the cases where the 'callbacks' variable
        is ``None``, a single function, or a list.
        """
        if isinstance(callbacks, list):
            for cb in callbacks:
                cb(*args, **kwargs)
            return
        if callbacks:
            callbacks(*args, **kwargs)

    # initialize Fisher
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = p.new_zeros(p.size())
        p.requires_grad = True
        p.retain_grad()
        print("done")

    # training
    for nbatch, batch in enumerate(train_loader):
        model.zero_grad()
        global_steps = len(train_loader) + nbatch
        os.environ['global_steps'] = str(global_steps)

        # record time
        data_in_time = time.time() - end_time

        # transfer data to GPU
        data_transfer_time = time.time()
        batch = to_cuda(batch)
        data_transfer_time = time.time() - data_transfer_time

        # forward
        forward_time = time.time()
        outputs, loss = model(*batch)
        loss = loss.mean()
        forward_time = time.time() - forward_time

        # backward
        backward_time = time.time()
        loss.backward()

        backward_time = time.time() - backward_time

        for n, p in model.named_parameters():
            fisher[n] += p.grad.data**2 / len(train_loader)

        batch_end_params = BatchEndParam(epoch=0, nbatch=nbatch, add_step=True, rank=rank,
                                         data_in_time=data_in_time, data_transfer_time=data_transfer_time,
                                         forward_time=forward_time, backward_time=backward_time,
                                         optimizer_time=0., metric_time=0.,
                                         eval_metric=None, locals=locals())
        _multiple_callbacks(batch_end_callbacks, batch_end_params)

    with open(os.path.join(config.EWC_STATS_PATH, "fisher"), "wb") as fisher_file:
        pickle.dump(fisher, fisher_file)
    torch.save(model.state_dict(), os.path.join(config.EWC_STATS_PATH, "opt_params"))
