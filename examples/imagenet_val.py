import argparse
import os
import random
import configparser
import collections

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import *

SEED = 123456

models = {'quant_mobilenet_v1': quant_mobilenet_v1,
          'quant_proxylessnas_mobile14': quant_proxylessnas_mobile14}


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--imagenet-dir', help='path to folder containing Imagenet val folder')
parser.add_argument('--model-cfg', type=str, help='Path to pretrained model .ini configuration file')
parser.add_argument('--output-dir', type=str, help='Path to export files')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=256, type=int, help='Minibatch size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--export', action='store_true')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    assert os.path.exists(args.model_cfg)
    cfg = configparser.ConfigParser()
    cfg.read(args.model_cfg)
    arch = cfg.get('MODEL', 'ARCH')

    model = models[arch](cfg)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = True

    pretrained_url = cfg.get('MODEL', 'PRETRAINED_URL')
    print("=> Loading checkpoint from:'{}'".format(pretrained_url))
    if args.gpu is None:
        loc = 'cpu'
    else:
        loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.hub.load_state_dict_from_url(pretrained_url, map_location=loc)
    model.load_state_dict(checkpoint, strict=True)

    valdir = os.path.join(args.imagenet_dir, 'val')
    mean = [float(cfg.get('PREPROCESS', 'MEAN_0')), float(cfg.get('PREPROCESS', 'MEAN_1')),
            float(cfg.get('PREPROCESS', 'MEAN_2'))]
    std = [float(cfg.get('PREPROCESS', 'STD_0')), float(cfg.get('PREPROCESS', 'STD_1')),
           float(cfg.get('PREPROCESS', 'STD_2'))]
    normalize = transforms.Normalize(mean=mean, std=std)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    validate(val_loader, model, args)


def validate(val_loader, model, args):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    def print_accuracy(top1, top5, prefix=''):
        print('{}Avg acc@1 {top1.avg:.3f} Avg acc@5 {top5.avg:.3f}'
              .format(prefix, top1=top1, top5=top5))

    model.eval()
    with torch.no_grad():
        num_batches = len(val_loader)
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if args.export:
                exp_input = images[0].unsqueeze(0)
                weight_list, threshold_list, config_list, int_input_list, int_acc_list = model.export(exp_input)
                exp_dicts = [dict(weight_list), dict(threshold_list)]
                output_dict = collections.defaultdict(list)
                for d in exp_dicts:
                    for k, v in d.items():
                        output_dict[k].append(v)
                for name_prefix, var_list in output_dict.items():
                    with open(os.path.join(args.output_dir, '{}.h'.format(name_prefix)), 'w') as f:
                        for var in var_list:
                            f.write(var)
                            f.write('\n')
                with open(os.path.join(args.output_dir, 'config.h'), 'w') as f:
                    for config_line in config_list:
                        f.write(config_line)
                for array_tuple in int_input_list:
                    np.save(os.path.join(args.output_dir, array_tuple[0] + '.npy'), array_tuple[1])
                for array_tuple in int_acc_list:
                    np.save(os.path.join(args.output_dir, array_tuple[0] + '.npy'), array_tuple[1])
                return

            output = model(images)
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            print_accuracy(top1, top5, '{}/{}: '.format(i, num_batches))
        print_accuracy(top1, top5, 'Total:')
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


if __name__ == '__main__':
    main()
