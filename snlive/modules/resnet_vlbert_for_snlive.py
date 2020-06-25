import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
root_path = os.path.abspath(os.getcwd())
if root_path not in sys.path:
    sys.path.append(root_path)
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertMVRCHeadTransform
from common.nlp.roberta import RobertaTokenizer

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)
        self.enable_cnn_reg_loss = config.NETWORK.ENABLE_CNN_REG_LOSS
        self.cnn_loss_top = config.NETWORK.CNN_LOSS_TOP
        if not config.NETWORK.BLIND:
            self.image_feature_extractor = FastRCNN(config,
                                                    average_pool=True,
                                                    final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                    enable_cnn_reg_loss=(self.enable_cnn_reg_loss and not self.cnn_loss_top))
            if config.NETWORK.VLBERT.object_word_embed_mode == 1:
                self.object_linguistic_embeddings = nn.Embedding(81, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 2:
                self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 3:
                self.object_linguistic_embeddings = None
            else:
                raise NotImplementedError

        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        if 'roberta' in config.NETWORK.BERT_MODEL_NAME:
            self.tokenizer = RobertaTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path

        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                           language_pretrained_model_path=language_pretrained_model_path)
        
        self.for_pretrain = False
        dim = config.NETWORK.VLBERT.hidden_size
        if config.NETWORK.SENTENCE.CLASSIFIER_TYPE == "2fc":
            self.sentence_cls = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.SENTENCE.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.SENTENCE.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.SENTENCE.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.SENTENCE.CLASSIFIER_HIDDEN_SIZE, 3),
            )
        elif config.NETWORK.SENTENCE.CLASSIFIER_TYPE == "1fc":
            self.sentence_cls = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.SENTENCE.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 3)
            )
        else:
            raise ValueError("Classifier type: {} not supported!".format(config.NETWORK.SENTENCE.CLASSIFIER_TYPE))

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        if not self.config.NETWORK.BLIND:
            self.image_feature_extractor.init_weight()
            if self.object_linguistic_embeddings is not None:
                self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        if not self.for_pretrain:
            for m in self.sentence_cls.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if (not self.config.NETWORK.BLIND) and self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        if self.config.NETWORK.BLIND:
            self.vlbert._module.visual_scale_text.requires_grad = False
            self.vlbert._module.visual_scale_object.requires_grad = False

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra dimensions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def prepare_text(self, sentence, mask):
        batch_size, max_len = sentence.shape
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        sep_pos = 1 + mask.sum(1, keepdim=True)
        input_ids = torch.zeros((batch_size, max_len + 2), dtype=sentence.dtype, device=sentence.device)
        input_ids[:, 0] = cls_id
        _batch_inds = torch.arange(sentence.shape[0], device=sentence.device)
        input_ids[_batch_inds, sep_pos] = sep_id
        input_ids[:, 1:-1] = sentence
        input_mask = input_ids > 0
        return input_ids, input_mask

    def train_forward(self,
                      images,
                      boxes,
                      hypothesis,
                      im_info,
                      label):
        ###########################################
        # visual feature extraction

        # Don't know what segments are for
        # segms = masks
        
        box_mask = (boxes[:, :, -1] > - 0.5)
        max_len = int(box_mask.sum(1).max().item())

        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len].type(torch.float32)

        # segms = segms[:, :max_len]
        if self.config.NETWORK.BLIND:
            obj_reps = {'obj_reps': boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.IMAGE_FINAL_DIM))}
        else:
            obj_reps = self.image_feature_extractor(images=images,
                                                    boxes=boxes,
                                                    box_mask=box_mask,
                                                    im_info=im_info,
                                                    classes=None,
                                                    segms=None)

        # For now no tags
        mask = (hypothesis > 0.5)
        sentence_label = label.view(-1)


        ############################################
        
        # prepare text
        text_input_ids, text_mask = self.prepare_text(hypothesis, mask)
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)

        # Add visual feature to text elements
        text_visual_embeddings = self._collect_obj_reps(text_input_ids.new_zeros(text_input_ids.size()),
                                                        obj_reps['obj_reps'])
        # Add textual feature to image element
        if self.config.NETWORK.BLIND:
            object_linguistic_embeddings = boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.VLBERT.hidden_size))
        else:
            object_linguistic_embeddings = self.object_linguistic_embeddings(
                boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long())
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT
        if self.config.NETWORK.NO_OBJ_ATTENTION or self.config.NETWORK.BLIND:
            box_mask.zero_()

        _, pooled_rep = self.vlbert(text_input_ids,
                                    text_token_type_ids,
                                    text_visual_embeddings,
                                    text_mask,
                                    object_vl_embeddings,
                                    box_mask,
                                    output_all_encoded_layers=False,
                                    output_text_and_object_separately=False,
                                    output_attention_probs=False)

        ###########################################
        outputs = {}
        
        # sentence classification
        sentence_logits = self.sentence_cls(pooled_rep).view((-1, 3))
        sentence_cls_loss = F.cross_entropy(sentence_logits, sentence_label)

        outputs.update({'sentence_label_logits': sentence_logits,
                        'sentence_label': sentence_label.long(),
                        'sentence_cls_loss': sentence_cls_loss})

        loss = sentence_cls_loss.mean()

        return outputs, loss

    def inference_forward(self,
                          images,
                          boxes,
                          hypothesis,
                          im_info):
        ###########################################
        # visual feature extraction

        # Don't know what segments are for
        # segms = masks

        box_mask = (boxes[:, :, -1] > - 0.5)
        max_len = int(box_mask.sum(1).max().item())

        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len].type(torch.float32)

        # segms = segms[:, :max_len]
        if self.config.NETWORK.BLIND:
            obj_reps = {'obj_reps': boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.IMAGE_FINAL_DIM))}
        else:
            obj_reps = self.image_feature_extractor(images=images,
                                                    boxes=boxes,
                                                    box_mask=box_mask,
                                                    im_info=im_info,
                                                    classes=None,
                                                    segms=None)

        # For now no tags
        mask = (hypothesis > 0.5)

        ############################################

        # prepare text
        text_input_ids, text_mask = self.prepare_text(hypothesis, mask)
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)

        # Add visual feature to text elements
        text_visual_embeddings = self._collect_obj_reps(text_input_ids.new_zeros(text_input_ids.size()),
                                                        obj_reps['obj_reps'])
        # Add textual feature to image element
        if self.config.NETWORK.BLIND:
            object_linguistic_embeddings = boxes.new_zeros((*boxes.shape[:-1], self.config.NETWORK.VLBERT.hidden_size))
        else:
            object_linguistic_embeddings = self.object_linguistic_embeddings(
                boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long())
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT
        if self.config.NETWORK.NO_OBJ_ATTENTION or self.config.NETWORK.BLIND:
            box_mask.zero_()

        _, pooled_rep = self.vlbert(text_input_ids,
                                    text_token_type_ids,
                                    text_visual_embeddings,
                                    text_mask,
                                    object_vl_embeddings,
                                    box_mask,
                                    output_all_encoded_layers=False,
                                    output_text_and_object_separately=False,
                                    output_attention_probs=False)

        ###########################################
        outputs = {}

        # sentence classification
        sentence_logits = self.sentence_cls(pooled_rep).view((-1, 3))

        outputs.update({'sentence_label_logits': sentence_logits})

        return outputs


def test_module():
    from snlive.function.config import config, update_config
    from snlive.data.build import make_dataloader
    os.chdir("../../")
    cfg_path = os.path.join('cfgs', 'snlive', 'base_4x16G_fp32.yaml')
    update_config(cfg_path)
    dataloader = make_dataloader(config, dataset=None, mode='train')
    module = ResNetVLBERT(config)
    for batch in dataloader:
        outputs, loss = module(*batch)
        print("batch done")


if __name__ == '__main__':
    test_module()
