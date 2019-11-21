"""
Based on example:
https://github.com/williamFalcon/pytorch-lightning/blob/master/pl_examples/full_examples/imagenet/imagenet_example.py
"""

from collections import OrderedDict
import random
import logging

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import numpy as np
from pytorch_lightning.root_module.root_module import LightningModule
from torch import optim
from torch.nn import CrossEntropyLoss

from .data.imagenet_dataloder import imagenet_train_loader, imagenet_val_loader
from .models import models_dict
from .smoothing import LabelSmoothing
from .hydra_logger import *
from .utils import filter_keys, state_dict_from_url_or_path, topk_accuracy, AverageMeter


optim_impl = {'SGD': optim.SGD}
scheduler_impl = {'COSINE': optim.lr_scheduler.CosineAnnealingLR}


class QuantImageNetClassification(LightningModule):

    def __init__(self, hparams):
        super(QuantImageNetClassification, self).__init__()
        self.hparams = hparams
        arch = self.hparams.model.ARCH
        self.model = models_dict[arch](self.hparams)
        self.configure_loss()
        self.load_pretrained_model()

        # Set random seeds
        self.set_random_seed(hparams.SEED)

        # Setup meters to track training averages per epoch
        self.train_loss_meter = AverageMeter()
        self.train_top1_meter = AverageMeter()
        self.train_top5_meter = AverageMeter()

        # Setup meters for logging val to cli
        self.val_loss_meter = AverageMeter()
        self.val_top1_meter = AverageMeter()
        self.val_top5_meter = AverageMeter()

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load_pretrained_model(self):
        try:
            pretrained_pth = self.hparams.model.PRETRAINED_PTH
        except KeyError:
            return
        if pretrained_pth is not None:
            self.model.load_state_dict(state_dict_from_url_or_path((pretrained_pth)), strict=True)
            logging.info('Loaded .pth at: {}'.format(pretrained_pth))

    def configure_ddp(self, model, device_ids):
        assert len(device_ids) == 1, 'Only 1 GPU per process supported'
        self.set_random_seed(self.hparams.SEED + device_ids[0])
        if torch.distributed.is_available():
            from .apex_lightning import LightningApexDistributedDataParallel as ApexDDP
            model = ApexDDP(model, device_ids=device_ids)
        else:
            raise Exception("Can't invoke DDP when torch.distributed is not available")
        return model

    def loss(self, output, target):
        if isinstance(output, tuple):  # supports multi-sample dropout
            loss = sum((self.__loss_fn(o, target) for o in output))
            loss = loss / len(output)
            output = sum(output) / len(output)
            return loss, output
        else:
            loss = self.__loss_fn(output, target)
            return loss, output

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        train_loss, output = self.loss(output, target)
        train_top1, train_top5 = topk_accuracy(output, target, topk=(1, 5))

        self.train_loss_meter.update(train_loss.detach())
        self.train_top1_meter.update(train_top1)
        self.train_top5_meter.update(train_top5)

        log_dict = OrderedDict({
            LOG_STAGE_LOG_KEY: LogStage.TRAIN_BATCH,
            EPOCH_LOG_KEY: self.current_epoch,
            BATCH_IDX_LOG_KEY: batch_idx,
            NUM_BATCHES_LOG_KEY: self.trainer.nb_training_batches,
            TRAIN_LOSS_METER: self.train_loss_meter,
            TRAIN_TOP1_METER: self.train_top1_meter,
            TRAIN_TOP5_METER: self.train_top5_meter
        })
        output = OrderedDict({
            'loss': train_loss,
            'log': log_dict
        })
        return output

    def on_epoch_start(self):
        self.train_loss_meter.reset()
        self.train_top1_meter.reset()
        self.train_top5_meter.reset()
        self.val_loss_meter.reset()
        self.val_top1_meter.reset()
        self.val_top5_meter.reset()

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        val_loss, output = self.loss(output, target)
        val_top1, val_top5 = topk_accuracy(output, target, topk=(1, 5))

        self.val_loss_meter.update(val_loss)
        self.val_top1_meter.update(val_top1)
        self.val_top5_meter.update(val_top5)

        log_dict = {LOG_STAGE_LOG_KEY: LogStage.VAL_BATCH,
                    EPOCH_LOG_KEY: self.current_epoch,
                    BATCH_IDX_LOG_KEY: batch_idx,
                    NUM_BATCHES_LOG_KEY: self.trainer.nb_val_batches,
                    VAL_LOSS_METER: self.val_loss_meter,
                    VAL_TOP1_METER: self.val_top1_meter,
                    VAL_TOP5_METER: self.val_top5_meter}
        self.logger.log_metrics(log_dict)
        return val_loss

    def validation_end(self, outputs):
        log_dict = {LOG_STAGE_LOG_KEY: LogStage.EPOCH,
                    EPOCH_LOG_KEY: self.current_epoch,
                    TRAIN_LOSS_METER: self.train_loss_meter,
                    TRAIN_TOP1_METER: self.train_top1_meter,
                    TRAIN_TOP5_METER: self.train_top5_meter,
                    VAL_LOSS_METER: self.val_loss_meter,
                    VAL_TOP1_METER: self.val_top1_meter,
                    VAL_TOP5_METER: self.val_top5_meter}

        result = {'log': log_dict, 'val_loss': log_dict[VAL_LOSS_METER].avg}
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def configure_optimizers(self):
        no_wd_params, wd_params = filter_keys(self.named_parameters(), self.hparams.NO_WD)
        optim_dict = [{'params': no_wd_params, 'weight_decay': 0.0},
                      {'params': wd_params, 'weight_decay': self.hparams.optim_conf.weight_decay}]
        optimizer = optim_impl[self.hparams.OPTIMIZER](optim_dict, **self.hparams.optim_conf)
        scheduler = scheduler_impl[self.hparams.SCHEDULER](optimizer, **self.hparams.scheduler_conf)
        return [optimizer], [scheduler]

    def configure_loss(self):
        if self.hparams.LABEL_SMOOTHING > 0.0:
            self.__loss_fn = LabelSmoothing(self.hparams.LABEL_SMOOTHING)
        else:
            self.__loss_fn = CrossEntropyLoss()

    def __dataloader(self, train):
        mean = list(self.hparams.preprocess.MEAN)
        std = list(self.hparams.preprocess.STD)

        def _worker_init_fn(id):
            seed = self.hparams.SEED + self.trainer.proc_rank + id
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if train:
            dataloader = imagenet_train_loader(batch_size=self.hparams.TRAIN_BATCH_SIZE,
                                               data_path=self.hparams.DATADIR,
                                               is_distributed=self.hparams.IS_DISTRIBUTED,
                                               mean=mean,
                                               std=std,
                                               workers=self.hparams.WORKERS,
                                               worker_init_fn=_worker_init_fn)
        else:
            dataloader = imagenet_val_loader(batch_size=self.hparams.VAL_BATCH_SIZE,
                                             data_path=self.hparams.DATADIR,
                                             mean=mean,
                                             std=std)
        return dataloader

    @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader(train=False)

    @pl.data_loader
    def test_dataloader(self):
        return self.__dataloader(train=False)



