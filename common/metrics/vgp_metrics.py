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
    def __init__(self, prefix_name="", allreduce=False, num_replicas=1):
        self.prefix.prefix = prefix_name
        super(Accuracy, self).__init__(prefix_name + 'Acc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            _filter = outputs[self.prefix + '_label'] != -1
            cls_logits = outputs[self.prefix + '_label_logits'][_filter]
            label = outputs[self.prefix + '_label'][_filter]
            if cls_logits.dim() == 1:
                self.sum_metric += float(((cls_logits > 0.).float() == label.float()).sum().item())
                self.num_inst += cls_logits.shape[0]
            if cls_logits.dim() == 2:
                self.sum_metric += float((cls_logits.argmax(dim=1) == label).sum().item())
                self.num_inst += cls_logits.shape[0]


def compute_metrics_sentence_level(metric, pred_labels, labels):
    labels = copy.deepcopy(labels)
    pred_labels = copy.deepcopy(pred_labels)
    if metric == "overall_accuracy":
        result = (pred_labels == labels).mean()
    elif metric == "easy_accuracy":
        labels[labels > 0] = 1
        pred_labels[pred_labels > 0] = 1
        result = (pred_labels == labels).mean()
    elif metric == "alignment_accuracy":
        mask = np.logical_and(pred_labels > 0, labels > 0)
        result = (pred_labels[mask] == labels[mask]).mean()
    else:
        print("The metric {} has not been implemented".format(metric))
    return result
