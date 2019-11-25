import os
from urllib.parse import urlparse

import torch


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


def state_dict_from_url_or_path(pretrained_pth):
    if os.path.exists(pretrained_pth):
        state_dict = torch.load(pretrained_pth, map_location='cpu')
    elif urlparse(pretrained_pth).netloc:  # validates the url
        state_dict = torch.hub.load_state_dict_from_url(pretrained_pth, map_location='cpu')
    else:
        raise Exception("Cant' load pretrained model at: {}".format(pretrained_pth))
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
