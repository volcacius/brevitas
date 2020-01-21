import argparse
import random

import hydra
import torch
import torch.backends.cudnn as cudnn
import torch.optim

from imagenet_classification.models import models_dict
from imagenet_classification.utils import topk_accuracy, state_dict_from_url_or_path, AverageMeter
from imagenet_classification.data.imagenet_dataloder import imagenet_val_loader
from imagenet_classification.models import layers
from imagenet_classification.models.layers.make_layer import MakeLayerWithDefaults

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')


@hydra.main(config_path='conf/inference_config.yaml')
def main(hparams):

    random.seed(hparams.SEED)
    torch.manual_seed(hparams.SEED)
    layers.with_defaults = MakeLayerWithDefaults(hparams.layers_defaults)
    model = models_dict[hparams.model.ARCH](hparams)

    assert hparams.model.PRETRAINED_MODEL is not None, 'Validation requires a pretrained model'
    state_dict = state_dict_from_url_or_path(hparams.model.PRETRAINED_MODEL)
    model.load_state_dict(state_dict, strict=True)

    if hparams.GPU is not None:
        assert isinstance(hparams.GPU, int), 'At most one GPU is supported for validation'
        torch.cuda.set_device(hparams.GPU)
        model = model.cuda(hparams.GPU)
        cudnn.benchmark = True

    val_loader = imagenet_val_loader(data_path=hparams.DATADIR,
                                     workers=hparams.WORKERS,
                                     batch_size=hparams.VAL_BATCH_SIZE,
                                     mean=hparams.preprocess.MEAN,
                                     std=hparams.preprocess.STD,
                                     resize_impl_type=hparams.preprocess.RESIZE)
    validate(val_loader, model, hparams.GPU)
    return


def print_final_accuracy(top1, top5):
    print('Avg val_top1 {:.4f} val_top5 {:.4f}'.format(top1.avg, top5.avg))


def print_accuracy(i, num_batches, top1, top5):
    print('{}/{}: val_top1: {:.4f} [{:.4f}] val_top5: {:.4f} [{:.4f}]'
          .format(i, num_batches, top1.val, top1.avg, top5.val, top5.avg))


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
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            print_accuracy(i, num_batches, top1, top5)
        print_final_accuracy(top1, top5)
    return


if __name__ == '__main__':
    main()

