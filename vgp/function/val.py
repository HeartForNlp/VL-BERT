from collections import namedtuple
import torch
from common.trainer import to_cuda


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch):
    net.eval()
    metrics.reset()
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch]
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        outputs = net(*datas)
        
        # handle labels whether phrases are classified or not
        if label.dim() == 2:
            outputs.update({'sentence_label': label.view(-1)})
        elif label.dim() == 3:
            sentence_label = label[:, 0, 0].view(-1)
            phrase_labels = label[:, :, 1]
            if "phrase_label_logits" in outputs.keys():
                phrase_labels = phrase_labels[phrase_labels > -1]
            outputs.update({"phrase_label": phrase_labels})
        
        metrics.update(outputs)
