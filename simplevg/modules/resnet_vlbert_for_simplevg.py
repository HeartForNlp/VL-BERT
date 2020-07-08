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
        self.supervise_attention = config.NETWORK.SUPERVISE_ATTENTION
        self.normalization = config.NETWORK.ATTENTION_NORM_METHOD
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
                torch.nn.Linear(config.NETWORK.SENTENCE.CLASSIFIER_HIDDEN_SIZE, 1),
            )
        elif config.NETWORK.SENTENCE.CLASSIFIER_TYPE == "1fc":
            self.sentence_cls = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.SENTENCE.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, 1)
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

    def prepare_text(self, sentence, mask, sentence_tags):
        batch_size, max_len = sentence.shape
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        sep_pos = 1 + mask.sum(1, keepdim=True)
        input_ids = torch.zeros((batch_size, max_len + 2), dtype=sentence.dtype, device=sentence.device)
        text_tags = sentence.new_zeros((batch_size, max_len + 2))
        input_ids[:, 0] = cls_id
        _batch_inds = torch.arange(sentence.shape[0], device=sentence.device)
        input_ids[_batch_inds, sep_pos] = sep_id
        input_ids[:, 1:-1] = sentence
        text_tags[:, 1:-1] = sentence_tags
        input_mask = input_ids > 0
        return input_ids, text_tags, input_mask

    def train_forward(self,
                      images,
                      boxes,
                      sentence,
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
        sentence_ids = sentence[:, :, 0]
        mask = (sentence[:, :, 0] > 0.5)
        sentence_tags = sentence[:, :, 1]
        sentence_label = label.view(-1)


        ############################################
        
        # prepare text
        text_input_ids, text_tags, text_mask = self.prepare_text(sentence_ids,
                                                                 mask,
                                                                 sentence_tags)
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)

        # Add visual feature to text elements
        if self.config.NETWORK.NO_GROUNDING:
            text_visual_embeddings = self._collect_obj_reps(text_tags.new_zeros(text_tags.size()), obj_reps['obj_reps'])
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])
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

        if self.supervise_attention:
            hidden_states_text, hidden_states_objects, pooled_rep, attention_probs = \
                self.vlbert(text_input_ids,
                            text_token_type_ids,
                            text_visual_embeddings,
                            text_mask,
                            object_vl_embeddings,
                            box_mask,
                            output_all_encoded_layers=False,
                            output_text_and_object_separately=True,
                            output_attention_probs=self.supervise_attention)
        else:
            hidden_states_text, hidden_states_objects, pooled_rep = self.vlbert(text_input_ids,
                                                                                text_token_type_ids,
                                                                                text_visual_embeddings,
                                                                                text_mask,
                                                                                object_vl_embeddings,
                                                                                box_mask,
                                                                                output_all_encoded_layers=False,
                                                                                output_text_and_object_separately=True,
                                                                                output_attention_probs=False)

        ###########################################
        outputs = {}
        
        # sentence classification
        sentence_logits = self.sentence_cls(pooled_rep).view(-1)
        sentence_cls_loss = F.binary_cross_entropy_with_logits(sentence_logits, sentence_label.type(torch.float32))

        outputs.update({'sentence_label_logits': sentence_logits,
                        'sentence_label': sentence_label.long(),
                        'sentence_cls_loss': sentence_cls_loss})

        # Handle attention supervision, suffix 1 refers to text-to-roi attention and suffix 2 refers to roi-to-text
        attention_loss_1 = 0.
        attention_loss_2 = 0.
        if self.supervise_attention:
            attention_loss_1, attention_loss_2 = get_attention_supervision_loss(attention_probs, text_tags, text_mask,
                                                                                box_mask,
                                                                                normalization=self.normalization)
            outputs.update({"attention_loss_1": attention_loss_1, "attention_loss_2": attention_loss_2})

        loss = sentence_cls_loss.mean() + self.config.NETWORK.ATTENTION_LOSS_WEIGHT * attention_loss_1 + \
               self.config.NETWORK.ATTENTION_LOSS_WEIGHT * attention_loss_2

        return outputs, loss

    def inference_forward(self,
                          images,
                          boxes,
                          sentence,
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
        sentence_ids = sentence[:, :, 0]
        mask = (sentence[:, :, 0] > 0.5)
        sentence_tags = sentence[:, :, 1]

        ############################################

        # prepare text
        text_input_ids, text_tags, text_mask = self.prepare_text(sentence_ids,
                                                                 mask,
                                                                 sentence_tags)
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)

        # Add visual feature to text elements
        if self.config.NETWORK.NO_GROUNDING:
            text_visual_embeddings = self._collect_obj_reps(text_tags.new_zeros(text_tags.size()), obj_reps['obj_reps'])
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])
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
        hidden_states_text, hidden_states_objects, pooled_rep = self.vlbert(text_input_ids,
                                                                            text_token_type_ids,
                                                                            text_visual_embeddings,
                                                                            text_mask,
                                                                            object_vl_embeddings,
                                                                            box_mask,
                                                                            output_all_encoded_layers=False,
                                                                            output_text_and_object_separately=True)

        ###########################################
        outputs = {}
        # sentence classification
        sentence_logits = self.sentence_cls(pooled_rep).view(-1)

        outputs.update({'sentence_label_logits': sentence_logits})

        return outputs


def get_attention_supervision_loss(attention_probs, text_tags, text_mask, box_mask, normalization):
    # suffix 1 refers to text-to-roi attention and suffix 2 refers to roi-to-text
    # reformat attention output by transposing batch and layer dimension
    attention_probs = [layer.unsqueeze(0) for layer in attention_probs]
    attention_probs = torch.cat(attention_probs).permute((1, 0, 2, 3, 4))
    attention_loss_1 = text_tags.new_zeros((len(attention_probs)), device=text_tags.device, dtype=torch.float32)
    attention_loss_2 = text_tags.new_zeros((len(attention_probs)), device=text_tags.device, dtype=torch.float32)
    grounded_words = torch.cat(((text_tags > 0),
                                text_tags.new_zeros((text_tags.size(0),
                                                     attention_probs[0].size(-1) - text_tags.size(1)),
                                                    dtype=torch.uint8)), dim=1)
    boxes_pos = torch.cat((box_mask.new_zeros((box_mask.size(0),
                                               attention_probs[0].size(-1) - 1 - box_mask.size(1)),
                                              dtype=torch.uint8), box_mask,
                           box_mask.new_zeros((box_mask.size(0), 1), dtype=torch.uint8)), dim=1)
    text_pos = torch.cat((text_mask, text_mask.new_zeros((text_mask.size(0),
                                                          attention_probs[0].size(-1) - text_mask.size(1)),
                                                         dtype=torch.uint8)), dim=1)
    n_layers = attention_probs.size(1)
    n_heads = attention_probs.size(2)
    n_grounded_boxes = torch.tensor([len(torch.unique(text_tags[i][text_tags[i] > 0]))
                                     for i in range(len(text_tags))]).sum()
    epsilon = 1e-6
    for i, attention in enumerate(attention_probs):
        if text_tags[i].sum().item() == 0:
            attention_loss_1[i] = 0.
            attention_loss_2[i] = 0.
            continue
        else:
            # Handle text-to-roi attention
            pred_attention_1 = attention[:, :, grounded_words[i]][:, :, :, boxes_pos[i]]
            # Flatten attention tensor with respect to layers and heads
            pred_attention_1 = pred_attention_1.view((-1, pred_attention_1.size(-1)))
            if normalization == "linear":
                norm_log_attention_1 = torch.log(epsilon +
                                                 pred_attention_1 / (pred_attention_1.sum(-1, keepdim=True) + epsilon))
            else:
                norm_log_attention_1 = F.log_softmax(pred_attention_1, dim=-1)
            attention_label_1 = text_tags[i][text_tags[i] > 0]
            # broadcast labels to same shape as attention
            attention_label_1 = attention_label_1.unsqueeze(0).unsqueeze(0).repeat((n_layers, n_heads, 1))
            # flatten labels with respect to layers and heads
            attention_label_1 = attention_label_1.view((-1))
            attention_loss_1[i] = F.nll_loss(norm_log_attention_1, attention_label_1, reduction="sum")

            # Handle roi-to-text attention
            grounded_boxes = torch.unique(attention_label_1)
            pred_attention_2 = attention[:, :, boxes_pos[i]][:, :, grounded_boxes][:, :, :, text_pos[i]]
            pred_attention_2 = pred_attention_2.view((-1, pred_attention_2.size(-1)))
            if normalization == "linear":
                norm_log_attention_2 = torch.log(epsilon +
                                                 pred_attention_2 / (pred_attention_2.sum(-1, keepdim=True) + epsilon))
            else:
                norm_log_attention_2 = F.log_softmax(pred_attention_2, dim=-1)
            attention_label_2 = text_tags[i].new_zeros((len(grounded_boxes), text_mask[i].sum()))
            attention_label_2[text_tags[i][text_mask[i]].unsqueeze(0).repeat(len(grounded_boxes), 1) ==
                              grounded_boxes.unsqueeze(1).repeat(1, text_mask[i].sum())] = 1
            # broadcast labels to same shape as attention
            attention_label_2 = attention_label_2.unsqueeze(0).unsqueeze(0).repeat((n_layers, n_heads, 1, 1))
            # flatten labels with respect to layers and heads
            attention_label_2 = attention_label_2.view((-1, attention_label_2.size(-1)))
            attention_loss_2[i] = F.binary_cross_entropy_with_logits(norm_log_attention_2, attention_label_2.type(torch.float32), reduction="sum") / text_mask[i].sum()

    # Average loss across all images and all grounded words
    if grounded_words.sum() != 0:
        attention_loss_1 = (attention_loss_1.sum() / (grounded_words.sum() * n_layers * n_heads))
        attention_loss_2 = (attention_loss_2.sum() / (n_grounded_boxes * n_layers * n_heads))
    else:
        attention_loss_1 = attention_loss_1.sum()
        attention_loss_2 = attention_loss_2.sum()
        
    return attention_loss_1, attention_loss_2


def test_module():
    from simplevg.function.config import config, update_config
    from simplevg.data.build import make_dataloader
    os.chdir("../../")
    cfg_path = os.path.join('cfgs', 'simplevg', 'base_4x16G_fp32.yaml')
    update_config(cfg_path)
    dataloader = make_dataloader(config, dataset=None, mode='train')
    module = ResNetVLBERT(config)
    for batch in dataloader:
        outputs, loss = module(*batch)
        print("batch done")


if __name__ == '__main__':
    test_module()
