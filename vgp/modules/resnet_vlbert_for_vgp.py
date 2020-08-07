import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import pickle
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
        self.align_caption_img = config.DATASET.ALIGN_CAPTION_IMG
        self.use_phrasal_paraphrases = config.DATASET.PHRASE_CLS
        self.supervise_attention = config.NETWORK.SUPERVISE_ATTENTION
        self.ewc_reg = config.NETWORK.EWC_REG
        self.importance_hparam = 0.
        if config.NETWORK.EWC_REG:
            self.fisher = pickle.load(open(config.NETWORK.FISHER_PATH, "rb"))
            self.pretrain_param = torch.load(config.NETWORK.PARAM_PRETRAIN)
            self.importance_hparam = config.NETWORK.EWC_IMPORTANCE
        if not config.NETWORK.BLIND:
            self.image_feature_extractor = FastRCNN(config,
                                                    average_pool=True,
                                                    final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                    enable_cnn_reg_loss=(self.enable_cnn_reg_loss
                                                                         and not self.cnn_loss_top))
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
        if self.align_caption_img:
            sentence_logits_shape = 3
        else:
            sentence_logits_shape = 1
        if config.NETWORK.SENTENCE.CLASSIFIER_TYPE == "2fc":
            self.sentence_cls = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.SENTENCE.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.SENTENCE.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.SENTENCE.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.SENTENCE.CLASSIFIER_HIDDEN_SIZE,
                                sentence_logits_shape),
            )
        elif config.NETWORK.SENTENCE.CLASSIFIER_TYPE == "1fc":
            self.sentence_cls = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.SENTENCE.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, sentence_logits_shape)
            )
        else:
            raise ValueError("Classifier type: {} not supported!".format(config.NETWORK.SENTENCE.CLASSIFIER_TYPE))

        if self.use_phrasal_paraphrases:
            if config.NETWORK.PHRASE.CLASSIFIER_TYPE == "2fc":
                self.phrasal_cls = torch.nn.Sequential(
                    torch.nn.Dropout(config.NETWORK.PHRASE.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(4*dim, config.NETWORK.PHRASE.CLASSIFIER_HIDDEN_SIZE),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(config.NETWORK.PHRASE.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(config.NETWORK.PHRASE.CLASSIFIER_HIDDEN_SIZE, 5),
                )
            elif config.NETWORK.PHRASE.CLASSIFIER_TYPE == "1fc":
                self.phrasal_cls = torch.nn.Sequential(
                    torch.nn.Dropout(config.NETWORK.PHRASE.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(4*dim, 5)
                )
            else:
                raise ValueError("Classifier type: {} not supported!".format(config.NETWORK.PHRASE.CLASSIFIER_TYPE))

        if self.supervise_attention == "indirect":
            if config.NETWORK.VG.CLASSIFIER_TYPE == "2fc":
                self.vg_cls = torch.nn.Sequential(
                    torch.nn.Dropout(config.NETWORK.VG.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(2*dim, config.NETWORK.VG.CLASSIFIER_HIDDEN_SIZE),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(config.NETWORK.VG.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(config.NETWORK.VG.CLASSIFIER_HIDDEN_SIZE, 1),
                )
            elif config.NETWORK.VG.CLASSIFIER_TYPE == "1fc":
                self.vg_cls = torch.nn.Sequential(
                    torch.nn.Dropout(config.NETWORK.VG.CLASSIFIER_DROPOUT, inplace=False),
                    torch.nn.Linear(2*dim, 1)
                )
            else:
                raise ValueError("Classifier type: {} not supported!".format(config.NETWORK.PHRASE.CLASSIFIER_TYPE))

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

    def prepare_text(self, sentence1, sentence2, mask1, mask2, sentence1_tags, sentence2_tags, phrase1_mask,
                     phrase2_mask):
        batch_size, max_len1 = sentence1.shape
        _, max_len2 = sentence2.shape
        max_len = (mask1.sum(1) + mask2.sum(1)).max() + 3
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        end_1 = 1 + mask1.sum(1, keepdim=True)
        end_2 = end_1 + 1 + mask2.sum(1, keepdim=True)
        input_ids = torch.zeros((batch_size, max_len), dtype=sentence1.dtype, device=sentence1.device)
        input_mask = torch.ones((batch_size, max_len), dtype=torch.uint8, device=sentence1.device)
        input_type_ids = torch.zeros((batch_size, max_len), dtype=sentence1.dtype, device=sentence1.device)
        text_tags = input_type_ids.new_zeros((batch_size, max_len))
        phr_mask = None
        grid_i, grid_k = torch.meshgrid(torch.arange(batch_size, device=sentence1.device),
                                        torch.arange(max_len, device=sentence1.device))

        input_mask[grid_k > end_2] = 0
        input_type_ids[(grid_k > end_1) & (grid_k <= end_2)] = 1
        input_mask1 = (grid_k > 0) & (grid_k < end_1)
        input_mask2 = (grid_k > end_1) & (grid_k < end_2)
        input_ids[:, 0] = cls_id
        input_ids[grid_k == end_1] = sep_id
        input_ids[grid_k == end_2] = sep_id
        input_ids[input_mask1] = sentence1[mask1]
        input_ids[input_mask2] = sentence2[mask2]
        text_tags[input_mask1] = sentence1_tags[mask1]
        text_tags[input_mask2] = sentence2_tags[mask2]
        if self.use_phrasal_paraphrases:
            phr_mask = phrase1_mask.new_zeros((batch_size, max_len, phrase1_mask.size(-1)))
            phr_mask[input_mask1] = phrase1_mask[mask1]
            phr_mask[input_mask2] = phrase2_mask[mask2]

            # add offsets so that every pair of phrases gets a unique id in the batch
            no_phr_mask = (phr_mask == 0)
            n_phr = torch.max(phr_mask, dim=1)[0]
            offsets = phr_mask.new_zeros((phr_mask.size(0)*phr_mask.size(-1)))
            offsets[1:] = torch.cumsum(n_phr.view(-1)[:-1], dim=0)
            offsets = offsets.view((phr_mask.size(0), phr_mask.size(-1)))
            phr_mask += offsets.unsqueeze(1)
            phr_mask[no_phr_mask] = 0

        return input_ids, input_type_ids, text_tags, input_mask, phr_mask

    def train_forward(self,
                      images,
                      boxes,
                      sentence1,
                      sentence2,
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
        sentence1_ids = sentence1[:, :, 0]
        mask1 = (sentence1[:, :, 0] > 0.5)
        sentence1_tags = sentence1[:, :, 1]

        sentence2_ids = sentence2[:, :, 0]
        mask2 = (sentence2[:, :, 0] > 0.5)
        sentence2_tags = sentence2[:, :, 1]

        if self.use_phrasal_paraphrases:
            phrase1_mask = sentence1[:, :, 2:]
            phrase2_mask = sentence2[:, :, 2:]
            sentence_label = label[:, 0, 0].view(-1)
            phrase_labels = label[:, :, 1]
        else:
            phrase1_mask, phrase2_mask = None, None
            sentence_label = label.view(-1)


        ############################################
        
        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, phrase_mask = self.prepare_text(sentence1_ids,
                                                                                                    sentence2_ids,
                                                                                                    mask1,
                                                                                                    mask2,
                                                                                                    sentence1_tags,
                                                                                                    sentence2_tags,
                                                                                                    phrase1_mask,
                                                                                                    phrase2_mask)

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

        if self.supervise_attention in ["direct", "semi-direct"]:
            hidden_states_text, hidden_states_objects, pooled_rep, attention_probs = \
                self.vlbert(text_input_ids,
                            text_token_type_ids,
                            text_visual_embeddings,
                            text_mask,
                            object_vl_embeddings,
                            box_mask,
                            output_all_encoded_layers=False,
                            output_text_and_object_separately=True,
                            output_attention_probs=True)
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
        sentence_logits = self.sentence_cls(pooled_rep)
        if self.align_caption_img:
            sentence_logits = sentence_logits.view((-1, 3))
            sentence_cls_loss = F.cross_entropy(sentence_logits, sentence_label)
        else:
            sentence_logits = sentence_logits.view(-1)
            sentence_cls_loss = F.binary_cross_entropy_with_logits(sentence_logits, sentence_label.type(torch.float32))

        outputs.update({'sentence_label_logits': sentence_logits,
                        'sentence_label': sentence_label.long(),
                        'sentence_cls_loss': sentence_cls_loss})

        # phrasal paraphrases classification (later)
        phrase_cls_loss = sentence_logits.new_zeros(())
        if self.use_phrasal_paraphrases:
            phrase_labels = phrase_labels.view((-1))
            phrase_cls_logits = sentence_logits.new_zeros((phrase_labels.size(0), 5))
            outputs.update({"phrase_label": phrase_labels, 
                            "phrase_label_logits": phrase_cls_logits, 
                            "phrase_cls_loss": phrase_cls_loss})
            if phrase_mask.max() > 0:
                logits = self.get_phrase_cls(hidden_states_text, phrase_mask, text_token_type_ids)
                phrase_cls_loss = F.cross_entropy(logits, phrase_labels[phrase_labels > -1], reduction="mean")
                phrase_cls_logits[(phrase_labels > -1)] = logits
                outputs.update({"phrase_label_logits": phrase_cls_logits,
                                "phrase_cls_loss": phrase_cls_loss})

        # Handle attention supervision, suffix 1 refers to text-to-roi attention and suffix 2 refers to roi-to-text
        attention_loss = 0.
        if self.supervise_attention in["direct", "semi-direct"]:
            use_raw = self.supervise_attention == "direct"
            attention_loss_1, attention_loss_2 = get_attention_supervision_loss(attention_probs, text_tags,
                                                                                text_mask, box_mask, use_raw=use_raw)
            outputs.update({"attention_loss_1": attention_loss_1, "attention_loss_2": attention_loss_2})
            attention_loss = attention_loss_1 + attention_loss_2

        elif self.supervise_attention == "indirect":
            attention_loss = self.get_indirect_vg_loss(hidden_states_text, hidden_states_objects, text_tags, text_mask,
                                                       box_mask)
            outputs.update({"vg_loss": attention_loss})

        # EWC regularization loss against catastrophic forgetting
        ewc_loss = 0.
        if self.ewc_reg:
            for n, p in self.named_parameters():
                name = "module." + n
                if name in self.fisher.keys():
                    ewc_loss += (self.fisher[name].to(p.device) *
                                 (p - self.pretrain_param[name].to(p.device))**2).sum()
            outputs.update({"ewc_loss": ewc_loss})

        loss = sentence_cls_loss.mean() + self.config.NETWORK.PHRASE_LOSS_WEIGHT * phrase_cls_loss + \
               self.config.NETWORK.ATTENTION_LOSS_WEIGHT * attention_loss + self.importance_hparam * ewc_loss

        return outputs, loss

    def inference_forward(self,
                          images,
                          boxes,
                          sentence1,
                          sentence2,
                          im_info):
        ###########################################
        # visual feature extraction

        # Don't know what segments are for
        # segms = masks

        # For now use all boxes
        box_mask = torch.ones(boxes[:, :, -1].size(), dtype=torch.uint8, device=boxes.device)

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
        sentence1_ids = sentence1[:, :, 0]
        mask1 = (sentence1[:, :, 0] > 0.5)
        sentence1_tags = sentence1[:, :, 1]
        sentence2_ids = sentence2[:, :, 0]
        mask2 = (sentence2[:, :, 0] > 0.5)
        sentence2_tags = sentence2[:, :, 1]

        if self.use_phrasal_paraphrases:
            phrase1_mask = sentence1[:, :, 2:]
            phrase2_mask = sentence2[:, :, 2:]
        else:
            phrase1_mask, phrase2_mask = None, None

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, phrase_mask = self.prepare_text(sentence1_ids,
                                                                                                   sentence2_ids,
                                                                                                   mask1,
                                                                                                   mask2,
                                                                                                   sentence1_tags,
                                                                                                   sentence2_tags,
                                                                                                   phrase1_mask,
                                                                                                   phrase2_mask)

        # Add visual feature to text elements
        if self.config.NETWORK.NO_GROUNDING:
            text_tags.zero_()
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
        sentence_logits = self.sentence_cls(pooled_rep)
        if self.align_caption_img:
            sentence_logits = sentence_logits.view((-1, 3))
        else:
            sentence_logits = sentence_logits.view(-1)
        outputs.update({'sentence_label_logits': sentence_logits})
        
        if self.use_phrasal_paraphrases:
            phrase_cls_logits = sentence_logits.new_zeros((1, 5)) + 100000
            outputs.update({"phrase_label_logits": phrase_cls_logits})
            if phrase_mask.max() > 0:
                phrase_cls_logits = self.get_phrase_cls(hidden_states_text, phrase_mask, text_token_type_ids)
                outputs.update({"phrase_label_logits": phrase_cls_logits})

        return outputs

    def get_phrase_cls(self, encoded_rep, phr_mask, token_type):
        n_pairs = phr_mask.max().item()
        phr_reps = encoded_rep.new_zeros((n_pairs, 2, encoded_rep.size(-1)))
        for i in range(n_pairs):
            # max pool representation of first phrase
            shaped_phr_mask = (phr_mask == i + 1).any(2)
            phr_reps[i, 0] = encoded_rep[(token_type == 0) & shaped_phr_mask].max(dim=0)[0]
            # max pool representation of second phrase
            phr_reps[i, 1] = encoded_rep[(token_type == 1) & shaped_phr_mask].max(dim=0)[0]
        final_phrases_rep = torch.cat((phr_reps[:, 0], phr_reps[:, 1], torch.abs(phr_reps[:, 0] - phr_reps[:, 1]),
                                       torch.mul(phr_reps[:, 0], phr_reps[:, 1])), dim=1)
        output_logits = self.phrasal_cls(final_phrases_rep)
        return output_logits

    def get_indirect_vg_loss(self, encoded_text, encoded_objects, text_tags, text_mask, box_mask):
        if text_tags.max() <= 0:
            return encoded_text.new_zeros((1)).sum()
        else:
            vg_inputs = []
            vg_labels = []
            indexes = find_phrases(text_tags)
            for i, k, length, tag in indexes:
                phrases_rep = encoded_text[i, k:k + length].max(dim=0)[
                    0]  # max pool encoding of the words in the phrase
                objects_reps = encoded_objects[i][box_mask[i]][1:]
                vg_inputs.append(
                    torch.cat((phrases_rep.unsqueeze(0).repeat(len(objects_reps), 1), objects_reps), dim=1))
                vg_lbl = text_tags.new_zeros((len(objects_reps)))
                vg_lbl[tag - 1] = 1
                vg_labels.append(vg_lbl)
            vg_inputs = torch.cat(vg_inputs, dim=0)
            vg_labels = torch.cat(vg_labels, dim=0)
            vg_logits = self.vg_cls(vg_inputs).view(-1)
            vg_loss = F.binary_cross_entropy_with_logits(vg_logits, vg_labels.float())
            return vg_loss


def get_attention_supervision_loss(raw_attention, text_tags, text_mask, box_mask, use_raw=True):
    # suffix 1 refers to text-to-roi attention and suffix 2 refers to roi-to-text
    # reformat attention output by transposing batch and layer dimension
    raw_attention = [layer.unsqueeze(0) for layer in raw_attention]
    raw_attention = torch.cat(raw_attention).permute((1, 0, 2, 3, 4))
    n_layers = raw_attention.size(1)
    n_heads = raw_attention.size(2)
    attention = raw_attention
    if not use_raw:
        attention = get_attention_rollout(raw_attention)
        n_heads = 1 # attention rollout is already averaged over the heads
    attention_loss_1 = text_tags.new_zeros((len(attention)), device=text_tags.device, dtype=torch.float32)
    attention_loss_2 = text_tags.new_zeros((len(attention)), device=text_tags.device, dtype=torch.float32)
    grounded_words = torch.cat(((text_tags > 0),
                                text_tags.new_zeros((text_tags.size(0),
                                                     attention[0].size(-1) - text_tags.size(1)),
                                                    dtype=torch.uint8)), dim=1)
    boxes_pos = torch.cat((box_mask.new_zeros((box_mask.size(0),
                                               attention[0].size(-1) - 1 - box_mask.size(1)),
                                              dtype=torch.uint8), box_mask,
                           box_mask.new_zeros((box_mask.size(0), 1), dtype=torch.uint8)), dim=1)
    text_pos = torch.cat((text_mask, text_mask.new_zeros((text_mask.size(0),
                                                          attention[0].size(-1) - text_mask.size(1)),
                                                         dtype=torch.uint8)), dim=1)
    n_grounded_boxes = torch.tensor([len(torch.unique(text_tags[i][text_tags[i] > 0]))
                                     for i in range(len(text_tags))]).sum()
    for i, attn in enumerate(attention):
        if text_tags[i].sum().item() == 0:
            attention_loss_1[i] = 0.
            attention_loss_2[i] = 0.
            continue
        else:
            # Handle text-to-roi attention
            attn_label_1 = text_tags[i][text_tags[i] > 0]
            # broadcast labels to same shape as attention
            attn_label_1 = attn_label_1.unsqueeze(0).unsqueeze(0).repeat((n_layers, n_heads, 1))
            # Select only attention from grounded word tokens to ROIs
            if use_raw:
                attn_1 = attn[:, :, grounded_words[i]][:, :, :, boxes_pos[i]]
            else:  # attention rollout is already averaged over the heads
                attn_1 = attn[:, grounded_words[i]][:, :, boxes_pos[i]]
            norm_log_attn_1 = F.log_softmax(attn_1, dim=-1)

            # flatten both attention and labels with respect to layers and heads
            norm_log_attn_1 = norm_log_attn_1.view((-1, attn_1.size(-1)))
            attn_label_1 = attn_label_1.view((-1))
            attention_loss_1[i] = F.nll_loss(norm_log_attn_1, attn_label_1, reduction="sum")

            # Handle roi-to-text attention
            grounded_boxes = torch.unique(attn_label_1)
            attn_label_2 = text_tags[i].new_zeros((len(grounded_boxes), text_mask[i].sum()))
            attn_label_2[text_tags[i][text_mask[i]].unsqueeze(0).repeat(len(grounded_boxes), 1)
                         == grounded_boxes.unsqueeze(1).repeat(1, text_mask[i].sum())] = 1
            # broadcast labels to same shape as attention
            attn_label_2 = attn_label_2.unsqueeze(0).unsqueeze(0).repeat((n_layers, n_heads, 1, 1))
            if use_raw:
                attn_2 = attn[:, :, boxes_pos[i]][:, :, grounded_boxes][:, :, :, text_pos[i]]
            else:  # attention rollout is already averaged over the heads
                attn_2 = attn[:, boxes_pos[i]][:, grounded_boxes][:, :, text_pos[i]]
            norm_log_attn_2 = F.log_softmax(attn_2, dim=-1)
            # flatten both attention and labels with respect to layers and heads
            norm_log_attn_2 = norm_log_attn_2.view((-1, attn_2.size(-1)))
            attn_label_2 = attn_label_2.view((-1, attn_label_2.size(-1)))
            attention_loss_2[i] = F.binary_cross_entropy_with_logits(norm_log_attn_2,
                                                                     attn_label_2.type(torch.float32),
                                                                     reduction="sum") / text_mask[i].sum()

    # Average loss across all images and all grounded words
    if grounded_words.sum() != 0:
        attention_loss_1 = (attention_loss_1.sum() / (grounded_words.sum() * n_layers * n_heads))
        attention_loss_2 = (attention_loss_2.sum() / (n_grounded_boxes * n_layers * n_heads))
    else:
        attention_loss_1 = attention_loss_1.sum()
        attention_loss_2 = attention_loss_2.sum()
        
    return attention_loss_1, attention_loss_2


def get_attention_rollout(raw_attention):
    # Average over heads
    avg_attn = raw_attention.mean(dim=2)
    # account for residual connections
    res_avg_attn = avg_attn + torch.eye(avg_attn.size(-1), dtype=avg_attn.dtype,
                                        device=avg_attn.device).unsqueeze(0).unsqueeze(0)
    res_avg_attn = res_avg_attn / res_avg_attn.sum(dim=-1).unsqueeze(-1)

    attn_rollout = res_avg_attn.new_zeros(res_avg_attn.size())
    n_layers = attn_rollout.size(1)
    # first layer is the same as direct attention
    attn_rollout[:, 0] = res_avg_attn[:, 0]
    for i in range(1, n_layers):
        attn_rollout[:, i] = torch.matmul(res_avg_attn[:, i], attn_rollout[:, i - 1])
    return attn_rollout



def find_phrases(text_tags):
    res = []
    length = 0
    value = None
    for i, caption in enumerate(text_tags):
        tags = caption.view(-1)
        for k, tag in enumerate(tags):
            if tag > 0:
                if value is None or tag == value:
                    length += 1
                else:
                    res.append((i, k, length, value))
                    length = 1
                value = tag
            else:
                if length > 0:
                    res.append((i, k, length, value))
                    length = 0
                    value = None
    return res


def test_module():
    from vgp.function.config import config, update_config
    from vgp.data.build import make_dataloader
    os.chdir("../../")
    cfg_path = os.path.join('cfgs', 'vgp', 'base_4x16G_fp32.yaml')
    update_config(cfg_path)
    dataloader = make_dataloader(config, dataset=None, mode='train')
    module = ResNetVLBERT(config)
    for batch in dataloader:
        outputs, loss = module(*batch)
        print("batch done")


if __name__ == '__main__':
    test_module()
