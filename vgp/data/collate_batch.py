import torch

from common.utils.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, phrase_cls=False, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind
        self.phrase_cls = phrase_cls

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
        max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        max_caption1_length = max([len(data[self.data_names.index('caption1')]) for data in batch])
        max_caption2_length = max([len(data[self.data_names.index('caption2')]) for data in batch])
        if self.phrase_cls and 'label' in self.data_names:
            max_label_length = max([len(data[self.data_names.index('label')]) for data in batch])
            max_n_phrases = max([data[self.data_names.index('caption1')].shape[1] for data in batch]) - 2

        for i, ibatch in enumerate(batch):
            out = {}
            image = ibatch[self.data_names.index('image')]
            out['image'] = clip_pad_images(image, max_shape, pad=0)

            boxes = ibatch[self.data_names.index('boxes')]
            out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-1)

            caption1 = ibatch[self.data_names.index('caption1')]
            caption2 = ibatch[self.data_names.index('caption2')]
            dimension_caption1 = 2 + max_n_phrases if self.phrase_cls else len(caption1[0])
            dimension_caption2 = 2 + max_n_phrases if self.phrase_cls else len(caption2[0])
            out['caption1'] = clip_pad_2d(caption1, (max_caption1_length, dimension_caption1), pad=0)
            out['caption2'] = clip_pad_2d(caption2, (max_caption2_length, dimension_caption2), pad=0)

            out['im_info'] = ibatch[self.data_names.index('im_info')]
            if 'label' in self.data_names:
                out['label'] = ibatch[self.data_names.index('label')]
                if self.phrase_cls:
                    label = ibatch[self.data_names.index('label')]
                    out['label'] = clip_pad_2d(label, (max_label_length, len(label[0])), pad=-1)
                

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for k, items in enumerate(zip(*batch)):
            if isinstance(items[0], torch.Tensor):
                out_tuple += (torch.stack(tuple(items), dim=0), )
            else:
                out_tuple += (list(items), )
        return out_tuple

