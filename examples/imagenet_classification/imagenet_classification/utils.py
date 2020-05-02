import os
from urllib.parse import urlparse
from enum import Enum

import torch

IGNORE_VALUE = -1


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            # Has to be a tensor cause .item() is called on it
            return torch.tensor(IGNORE_VALUE)
        else:
            return self.sum / self.count


def filter_keys(dict_to_filter, filters, return_dict=False):
    include_dict = {}
    exclude_dict = {}
    for n, v in dict_to_filter:
        include = filters is not None and any([f in n for f in filters])
        if include:
            include_dict[n] = v
        else:
            exclude_dict[n] = v
    if return_dict:
        return include_dict, exclude_dict
    else:
        return include_dict.values(), exclude_dict.values()

def lowercase_keys(d):
    return {k.lower(): v for k, v in d.items()}


def state_dict_from_url_or_path(pretrained_model, load_from_ema):
    if os.path.exists(pretrained_model):
        if pretrained_model.lower().endswith('.pth'):
            state_dict = torch.load(pretrained_model, map_location='cpu')
        else:
            d = 'ema_state_dict' if load_from_ema else 'state_dict'
            state_dict = torch.load(pretrained_model, map_location='cpu')[d]
    elif urlparse(pretrained_model).netloc:  # validates the url
        state_dict = torch.hub.load_state_dict_from_url(pretrained_model, map_location='cpu')
    else:
        raise Exception("Can't load pretrained model at: {}".format(pretrained_model))
    return state_dict


def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res


class AutoName(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
         return name

    def __str__(self):
        return self.value


class MissingOptionalDependency:

    def __init__(self, e):
        self.e = e

    def __call__(self, *args, **kwargs):
        raise self.e
