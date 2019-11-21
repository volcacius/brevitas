import argparse
import random

import hydra
import torch
import torch.backends.cudnn as cudnn
import torch.optim

from imagenet_classification.models import models_dict
from imagenet_classification.utils import topk_accuracy, state_dict_from_url_or_path, AverageMeter
from imagenet_classification.data.imagenet_dataloder import imagenet_val_loader

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')


@hydra.main(config_path='conf/inference_config.yaml')
def main(hparams):

    random.seed(hparams.SEED)
    torch.manual_seed(hparams.SEED)

    model = models_dict[hparams.model.ARCH](hparams)

    assert hparams.model.PRETRAINED_PTH is not None, 'Validation requires a pretrained model'
    state_dict = state_dict_from_url_or_path(hparams.model.PRETRAINED_PTH)
    model.load_state_dict(state_dict, strict=True)

    if hparams.GPUS is not None:
        assert isinstance(hparams.GPUS, int), 'At most one GPU is supported for validation'
        torch.cuda.set_device(hparams.GPUS)
        model = model.cuda(hparams.GPUS)
        cudnn.benchmark = True

    val_loader = imagenet_val_loader(data_path=hparams.DATADIR,
                                     batch_size=hparams.VAL_BATCH_SIZE,
                                     mean=hparams.preprocess.MEAN,
                                     std=hparams.preprocess.STD)
    validate(val_loader, model, hparams.GPUS)
    return


def print_accuracy(top1, top5, prefix=''):
    print('{}Avg acc@1 {top1.avg:.3f} Avg acc@5 {top5.avg:.3f}'.format(prefix, top1=top1, top5=top5))


def validate(val_loader, model, gpu):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        num_batches = len(val_loader)
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = model(images)
            # measure accuracy
            acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            print_accuracy(top1, top5, '{}/{}: '.format(i, num_batches))
        print_accuracy(top1, top5, 'Total:')
    return


if __name__ == '__main__':
    main()

