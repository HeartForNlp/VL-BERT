import os
import time
import jsonlines
from PIL import Image
from copy import deepcopy
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import sys
root_path = os.path.abspath(os.getcwd())
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer, BasicTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.nlp.roberta import RobertaTokenizer



class SnliVEDataset(Dataset):
    def __init__(self, flickr_root, snlive_root, annotations_file, image_set, roi_set, transform=None, test_mode=False,
                 basic_tokenizer=None, tokenizer=None, pretrained_model_name=None, add_image_as_a_box=True, **kwargs):
        """
        Visual Grounded Dataset

        :param image_set: image folder name, e.g., 'vcr1images'
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param kwargs:
        """
        super(SnliVEDataset, self).__init__()

        self.annotations_file = os.path.join(snlive_root, annotations_file)
        self.image_set = os.path.join(flickr_root, image_set)
        self.roi_set = os.path.join(flickr_root, roi_set)
        self.transform = transform
        self.test_mode = test_mode
        self.add_image_as_a_box = add_image_as_a_box
        self.basic_tokenizer = basic_tokenizer if basic_tokenizer is not None \
            else BasicTokenizer(do_lower_case=True)
        if tokenizer is None:
            if pretrained_model_name is None:
                pretrained_model_name = 'bert-base-uncased'
            if 'roberta' in pretrained_model_name:
                tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
            else:
                tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer = tokenizer

        self.database = self.load_captions()

    def load_captions(self):
        database = []

        with jsonlines.open(self.annotations_file) as jsonl_file:
            for line in jsonl_file:
                #print(line.keys())
                # => Flikr30kID can be used to find corresponding Flickr30k image premise
                Flikr30kID = str(line['Flickr30K_ID'])
                # =>  gold_label is the label assigned by the majority label in annotator_labels (at least 3 out of 5),
                # If such a consensus is not reached, the gold label is marked as "-",
                # which are already filtered out from our SNLI-VE dataset
                label = str(line['gold_label'])
                # => hypothesis is the text hypothesis
                hypothesis = str(line['sentence2'])
                database.append({"img_id": Flikr30kID, "hypothesis": hypothesis, "label": encode_label(label)})
        # ignore or not find cached database, reload it from annotation file
        print('loading database from {} and creating pairs...'.format(self.annotations_file))
        tic = time.time()


        print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    def __getitem__(self, index):
        idb = deepcopy(self.database[index])

        # Load image and regions of interest
        img_id = idb['img_id']
        image = self._load_image(img_id)
        boxes = self._load_roi(img_id)
        w0, h0 = image.size

        boxes = torch.tensor(boxes)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box.type(torch.float32), boxes.type(torch.float32)), dim=0)

        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, _, im_info = self.transform(image, boxes, None, im_info)

        # clamp boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w0 - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h0 - 1)

        # Tokenize and align tokens with visual boxes
        hypo_tokens = self.tokenizer.tokenize(idb["hypothesis"])

        # Convert token to ids and concatenate visual grounding
        hypo_ids = torch.as_tensor(self.tokenizer.convert_tokens_to_ids(hypo_tokens))

        # Load label
        label = torch.as_tensor(int(idb['label'])) if not self.test_mode else None

        if not self.test_mode:
            outputs = (image, boxes, hypo_ids, im_info, label)
        else:
            outputs = (image, boxes, hypo_ids, im_info)

        return outputs

    def __len__(self):
        return len(self.database)

    def _load_image(self, img_id):
        path = os.path.join(self.image_set, img_id + ".jpg")
        return Image.open(path)

    def _load_roi(self, img_id):
        path = os.path.join(self.roi_set, img_id + '.xml')
        boxes = []
        parsed_tree = ET.parse(path)
        root = parsed_tree.getroot()
        for obj in root[2:]:
            if obj[1].tag == 'bndbox':
                dimensions = {obj[1][dim].tag: int(obj[1][dim].text) for dim in range(4)}
                boxes.append([dimensions['xmin'], dimensions['ymin'], dimensions['xmax'], dimensions['ymax']])
        return boxes

    @property
    def data_names(self):
        if not self.test_mode:
            data_names = ['image', 'boxes', 'hypothesis', 'im_info', 'label']
        else:
            data_names = ['image', 'boxes', 'hypothesis', 'im_info']

        return data_names


def encode_label(label):
    if label == "contradiction":
        return 0
    elif label == "entailment":
        return 1
    elif label == "neutral":
        return 2
    else:
        assert False, print("label {} not recognized".format(label))


def test_snlive():
    print(os.getcwd())
    flickr_root = "data/vgp/"
    snlive_root = "data/snli-ve/snli_1.0/"
    annotations_file = "snli_ve_train.jsonl"
    image_set = "flickr30k-images"
    roi_set = "Annotations"
    dataset = SnliVEDataset(flickr_root=flickr_root, snlive_root=snlive_root, annotations_file=annotations_file,
                            roi_set=roi_set, image_set=image_set)
    print(dataset.__getitem__(3))
    

if __name__ == "__main__":
    test_snlive()

