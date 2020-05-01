"""
Based on example:
https://github.com/williamFalcon/pytorch-lightning/blob/master/pl_examples/full_examples/imagenet/imagenet_example.py
"""

import os
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist

from pytorch_lightning.root_module.root_module import LightningModule
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from brevitas import config

import brevitas.nn as qnn

from .utils import MissingOptionalDependency

try:
    from apex.optimizers import FusedAdam
except Exception as e:
    FusedAdam = MissingOptionalDependency(e)

try:
    from apex.optimizers import FusedNovoGrad
except Exception as e:
    FusedNovoGrad = MissingOptionalDependency(e)

try:
    from timm.optim.rmsprop_tf import RMSpropTF
except Exception as e:
    RMSpropTF = MissingOptionalDependency(e)

from .data.imagenet_dataloder import imagenet_train_loader, imagenet_val_loader
from .models import models_dict
from .models.layers.make_layer import MakeLayerWithDefaults
from .models import layers
from .smoothing import LabelSmoothing
from .reg import MaxL2ScalingReg
from .hydra_logger import *
from .utils import filter_keys, state_dict_from_url_or_path, topk_accuracy, AverageMeter, lowercase_keys

optim_impl = {
    'NAG': SGD,
    'SGD': SGD,
    'ADAM': FusedAdam,
    'NOVOGRAD': FusedNovoGrad,
    'RMSPROPTF': RMSpropTF}

scheduler_impl = {
    'COSINE': lambda optim, t_max, lr_min: CosineAnnealingLR(optim, T_max=t_max, eta_min=lr_min),
    'MULTISTEP': MultiStepLR}


class QuantImageNetClassification(LightningModule):

    def __init__(self, hparams):
        super(QuantImageNetClassification, self).__init__()
        self.hparams = hparams
        self.configure_brevitas()
        self.configure_layers_defaults()
        self.configure_model()
        self.configure_ema()
        self.configure_loss()
        self.load_pretrained_model()
        self.set_random_seed(hparams.SEED)
        self.configure_meters()

    def configure_brevitas(self):
        batches_per_step = self.hparams.TRAIN_BATCH_SIZE * int(os.environ.get('WORLD_SIZE', 1))
        steps_per_epoch = self.hparams.DATASET_TRAIN_SIZE // batches_per_step
        config.TOTAL_NUM_STEPS = steps_per_epoch * self.hparams.EPOCHS

    def configure_meters(self):
        self.train_loss_meter = AverageMeter()
        self.train_top1_meter = AverageMeter()
        self.train_top5_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        self.val_top1_meter = AverageMeter()
        self.val_top5_meter = AverageMeter()
        self.val_loss_ema_meter = AverageMeter()
        self.val_top1_ema_meter = AverageMeter()
        self.val_top5_ema_meter = AverageMeter()

    def configure_layers_defaults(self):
        layers.with_defaults = MakeLayerWithDefaults(self.hparams.layers_defaults)

    def configure_model(self):
        arch = self.hparams.model.ARCH
        self.model = models_dict[arch](self.hparams)

    def configure_ema(self):
        arch = self.hparams.model.ARCH
        self.ema = models_dict[arch](self.hparams)
        self.ema.eval()
        self.ema_first_batch = True
        self.ema_first_epoch = True
        for p in self.ema.parameters():
            p.requires_grad_(False)
        for name, mod in self.ema.named_modules():
            if hasattr(mod, 'quant_buffers_eval'):
                mod.quant_buffers_eval = True

    def configure_optimizers(self):
        no_wd_params, wd_params = filter_keys(self.named_parameters(), self.hparams.NO_WD)
        optim_dict = [
            {'params': no_wd_params, 'weight_decay': 0.0},
            {'params': wd_params, 'weight_decay': self.hparams.optim_conf.WEIGHT_DECAY}]
        optimizer = optim_impl[self.hparams.OPTIMIZER](
            optim_dict, **lowercase_keys(self.hparams.optim_conf))
        scheduler = scheduler_impl[self.hparams.SCHEDULER](
            optimizer, **lowercase_keys(self.hparams.scheduler_conf))
        return [optimizer], [scheduler]

    def configure_loss(self):
        if self.hparams.LABEL_SMOOTHING > 0.0:
            self.__loss_fn = LabelSmoothing(self.hparams.LABEL_SMOOTHING)
        else:
            self.__loss_fn = CrossEntropyLoss()
        if self.hparams.MAX_L2_SCALING_REG > 0.0:
            self.max_l2_scaling_reg = MaxL2ScalingReg(self.hparams.MAX_L2_SCALING_REG)
        else:
            self.max_l2_scaling_reg = None

    def configure_ddp(self, model, device_ids):
        assert len(device_ids) == 1, 'Only 1 GPU per process supported'
        if torch.distributed.is_available():
            from .pl_overrides.pl_apex import LightningApexDistributedDataParallel as ApexDDP
            model = ApexDDP(model)
        else:
            raise Exception("Can't invoke DDP when torch.distributed is not available")
        return model

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load_pretrained_model(self):
        try:
            pretrained_model = self.hparams.model.PRETRAINED_MODEL
        except KeyError:
            return
        if pretrained_model is not None:
            self.model.load_state_dict(
                state_dict_from_url_or_path(pretrained_model, load_ema=self.hparams.LOAD_EMA),
                strict=self.hparams.STRICT)
            logging.info('Loaded pretrained model at: {}'.format(pretrained_model))

    def on_save_checkpoint(self, checkpoint):
        # Split model. from ema.
        state_dict = checkpoint['state_dict']
        model_state_dict = {}
        ema_state_dict = {}
        for k in state_dict.keys():
            if k.startswith('model.'):
                model_state_dict[k[len('model.'):]] = state_dict[k]
            elif k.startswith('ema.'):
                ema_state_dict[k[len('ema.'):]] = state_dict[k]
            else:
                model_state_dict[k] = state_dict[k]
                ema_state_dict[k] = state_dict[k]
        checkpoint['state_dict'] = model_state_dict
        checkpoint['ema_state_dict'] = ema_state_dict

    def loss(self, output, target):
        if isinstance(output, tuple):  # supports multi-sample dropout
            loss = sum((self.__loss_fn(o, target) for o in output))
            loss = loss / len(output)
            output = sum(output) / len(output)
        else:
            loss = self.__loss_fn(output, target)
        if self.max_l2_scaling_reg is not None:
            loss += self.max_l2_scaling_reg.loss(self.model)
        return loss, output

    def optimizer_step(
            self,
            current_epoch,
            batch_nb,
            optimizer,
            optimizer_i,
            second_order_closure=None):

        # LR warmup batch-by-batch
        if current_epoch < self.hparams.WARMUP_EPOCHS:
            warmup_batches = self.trainer.nb_training_batches * self.hparams.WARMUP_EPOCHS
            lr_scale = min(1., float(self.trainer.global_step + 1) / warmup_batches)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.optim_conf.LR

        # update params
        optimizer.step()
        optimizer.zero_grad()

        #update EMA
        self.update_ema_batch()

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        train_loss, output = self.loss(output, target)
        train_top1, train_top5 = topk_accuracy(output, target, topk=(1, 5))

        self.train_loss_meter.update(train_loss.detach())
        self.train_top1_meter.update(train_top1.detach())
        self.train_top5_meter.update(train_top5.detach())

        log_dict = OrderedDict({
            LOG_STAGE_LOG_KEY: LogStage.TRAIN_BATCH,
            EPOCH_LOG_KEY: self.current_epoch,
            BATCH_IDX_LOG_KEY: batch_idx,
            NUM_BATCHES_LOG_KEY: self.trainer.nb_training_batches,
            TRAIN_LOSS_METER: self.train_loss_meter,
            TRAIN_TOP1_METER: self.train_top1_meter,
            TRAIN_TOP5_METER: self.train_top5_meter})
        output = OrderedDict({
            'loss': train_loss,
            'log': log_dict})
        return output

    def update_ema_batch(self):
        with torch.no_grad():
            ema_state_dict = self.ema.state_dict()
            for k, new_value in self.model.state_dict().items():
                ema_value = ema_state_dict[k].detach() # this is a pointer to the tensor in the state dict, not a copy
                if self.ema_first_batch:
                    ema_value.copy_(new_value)
                    self.ema_first_batch = False
                else:
                    ema_value.copy_(ema_value * self.hparams.EMA_DECAY + (1.0 - self.hparams.EMA_DECAY) * new_value)

    def update_ema_epoch(self):
        with torch.no_grad():
            ema_state_dict = self.ema.state_dict()
            k_suffix = 'quant_weight_buffer'
            for k, value in ema_state_dict.items():
                if k_suffix in k:
                    quant_weight_buffer = ema_state_dict[k]
                    weight_k = k[:-len(k_suffix)] + 'weight'
                    ema_state_dict[weight_k].copy_(quant_weight_buffer.detach())
            if self.hparams.RELOAD_EMA_EPOCH:
                self.model.load_state_dict(ema_state_dict)

    def on_epoch_start(self):
        # Reset loggers
        self.train_loss_meter.reset()
        self.train_top1_meter.reset()
        self.train_top5_meter.reset()
        self.val_loss_meter.reset()
        self.val_top1_meter.reset()
        self.val_top5_meter.reset()
        self.val_loss_ema_meter.reset()
        self.val_top1_ema_meter.reset()
        self.val_top5_ema_meter.reset()
        # Reload ema weights into model if enabled, after the first epoch
        if self.ema_first_epoch:
            self.ema_first_epoch = False
        else:
            self.update_ema_epoch()


    def validation_step(self, batch, batch_idx):
        images, target = batch

        output = self.model(images)
        val_loss, output = self.loss(output, target)
        val_top1, val_top5 = topk_accuracy(output, target, topk=(1, 5))

        output_ema = self.ema(images)
        val_loss_ema, output_ema = self.loss(output_ema, target)
        val_top1_ema, val_top5_ema = topk_accuracy(output_ema, target, topk=(1, 5))

        self.val_loss_meter.update(val_loss.detach(), images.size(0))
        self.val_top1_meter.update(val_top1.detach(), images.size(0))
        self.val_top5_meter.update(val_top5.detach(), images.size(0))

        self.val_loss_ema_meter.update(val_loss_ema.detach(), images.size(0))
        self.val_top1_ema_meter.update(val_top1_ema.detach(), images.size(0))
        self.val_top5_ema_meter.update(val_top5_ema.detach(), images.size(0))

        log_dict = {
            LOG_STAGE_LOG_KEY: LogStage.VAL_BATCH,
            EPOCH_LOG_KEY: self.current_epoch,
            BATCH_IDX_LOG_KEY: batch_idx,
            NUM_BATCHES_LOG_KEY: self.trainer.nb_val_batches,
            VAL_LOSS_METER: self.val_loss_meter,
            VAL_TOP1_METER: self.val_top1_meter,
            VAL_TOP5_METER: self.val_top5_meter,
            VAL_LOSS_EMA_METER: self.val_loss_ema_meter,
            VAL_TOP1_EMA_METER: self.val_top1_ema_meter,
            VAL_TOP5_EMA_METER: self.val_top5_ema_meter}
        self.logger.log_metrics(log_dict)
        return val_loss

    def validation_end(self, outputs):
        log_dict = {
            LOG_STAGE_LOG_KEY: LogStage.EPOCH,
            EPOCH_LOG_KEY: self.current_epoch,
            TRAIN_LOSS_METER: self.train_loss_meter,
            TRAIN_TOP1_METER: self.train_top1_meter,
            TRAIN_TOP5_METER: self.train_top5_meter,
            VAL_LOSS_METER: self.val_loss_meter,
            VAL_TOP1_METER: self.val_top1_meter,
            VAL_TOP5_METER: self.val_top5_meter,
            VAL_LOSS_EMA_METER: self.val_loss_ema_meter,
            VAL_TOP1_EMA_METER: self.val_top1_ema_meter,
            VAL_TOP5_EMA_METER: self.val_top5_ema_meter}

        result = {
            'log': log_dict,
            'val_top1': log_dict[VAL_TOP1_METER].avg,
            'val_loss': log_dict[VAL_LOSS_METER].avg,
            'val_top1_ema': log_dict[VAL_TOP1_EMA_METER].avg,
            'val_loss_ema': log_dict[VAL_LOSS_EMA_METER].avg}
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def __dataloader(self, train):
        mean = list(self.hparams.preprocess.MEAN)
        std = list(self.hparams.preprocess.STD)

        if train:
            dataloader = imagenet_train_loader(
                batch_size=self.hparams.TRAIN_BATCH_SIZE,
                data_path=self.hparams.DATADIR,
                is_distributed=self.hparams.IS_DISTRIBUTED,
                mean=mean,
                std=std,
                workers=self.hparams.WORKERS,
                worker_init_fn=None,
                crop_size=self.hparams.preprocess.CROP_SIZE,
                resize_impl_type=self.hparams.preprocess.RESIZE_IMPL_TYPE)
        else:
            dataloader = imagenet_val_loader(
                batch_size=self.hparams.VAL_BATCH_SIZE,
                workers=self.hparams.WORKERS,
                data_path=self.hparams.DATADIR,
                mean=mean,
                std=std,
                resize_ratio=self.hparams.preprocess.RESIZE_RATIO,
                crop_size=self.hparams.preprocess.CROP_SIZE,
                resize_impl_type=self.hparams.preprocess.RESIZE_IMPL_TYPE)
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



