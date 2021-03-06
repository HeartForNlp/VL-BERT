import torch
from .eval_metric import EvalMetric
import numpy as np
import copy


class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class Accuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(Accuracy, self).__init__('Acc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            _filter = outputs['sentence_label'] != -1
            cls_logits = outputs['sentence_label_logits'][_filter]
            label = outputs['sentence_label'][_filter]
            self.sum_metric += float(((cls_logits > 0.).float() == label.float()).sum().item())
            self.num_inst += cls_logits.shape[0]
