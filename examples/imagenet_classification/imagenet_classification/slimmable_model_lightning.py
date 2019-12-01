from apex import amp
import torch.nn.functional as F
from torch import nn, bmm, mean

from collections import OrderedDict

from brevitas.core.bit_width import BitWidthConst

from .hydra_logger import *
from .utils import topk_accuracy, AverageMeter
from .model_lighting import QuantImageNetClassification

BW = '_{}b'
TRAIN_LOSS_BW_METER = TRAIN_ + LOSS_LOG_KEY + BW + METER_SUFFIX
TRAIN_TOP1_BW_METER = TRAIN_ + TOP1_LOG_KEY + BW + METER_SUFFIX
TRAIN_TOP5_BW_METER = TRAIN_ + TOP5_LOG_KEY + BW + METER_SUFFIX

VAL_LOSS_BW_METER = VAL_ + LOSS_LOG_KEY + BW + METER_SUFFIX
VAL_TOP1_BW_METER = VAL_ + TOP1_LOG_KEY + BW + METER_SUFFIX
VAL_TOP5_BW_METER = VAL_ + TOP5_LOG_KEY + BW + METER_SUFFIX


class CrossEntropyLossSoft(nn.Module):
    """ https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py """
    def forward(self, output, target):
        output_log_prob = F.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -bmm(target, output_log_prob)
        return mean(cross_entropy_loss)


class SlimmableWeightBitWidth(QuantImageNetClassification):

    def configure_model(self):
        super().configure_model()
        self.slimmable_bw = {}
        for n, m in self.model.named_modules():
            if isinstance(m, BitWidthConst):
                if m.bit_width == self.hparams.model.WEIGHT_BIT_WIDTH:
                    self.slimmable_bw[n] = True
                else:
                    self.slimmable_bw[n] = False

    def configure_loss(self):
        super().configure_loss()
        self.soft_loss = CrossEntropyLossSoft()

    def configure_meters(self):
        for bw in self.hparams.slimmable.WEIGHT_BIT_WIDTH:
            setattr(self, TRAIN_LOSS_BW_METER.format(bw), AverageMeter())
            setattr(self, TRAIN_TOP1_BW_METER.format(bw), AverageMeter())
            setattr(self, TRAIN_TOP5_BW_METER.format(bw), AverageMeter())
            setattr(self, VAL_LOSS_BW_METER.format(bw), AverageMeter())
            setattr(self, VAL_TOP1_BW_METER.format(bw), AverageMeter())
            setattr(self, VAL_TOP5_BW_METER.format(bw), AverageMeter())

    def backward(self, use_amp, loss, optimizer, is_substep=False):
        if is_substep and use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif is_substep and not use_amp:
            loss.backward()
        else:
            pass

    def set_bw(self, bw):
        for n, m in self.model.named_modules():
            if n in self.slimmable_bw and self.slimmable_bw[n]:
                m.bit_width = int(bw)

    def training_substep(self, bw, images, target, soft_target, loss_fn):
        self.set_bw(bw)
        output = self.model(images)
        if soft_target is not None:
            train_loss, output = self.multisample_loss(output, soft_target, loss_fn)
        else:
            train_loss, output = self.multisample_loss(output, target, loss_fn)
        train_top1, train_top5 = topk_accuracy(output, target, topk=(1, 5))
        getattr(self, TRAIN_LOSS_BW_METER.format(bw)).update(train_loss.detach())
        getattr(self, TRAIN_TOP1_BW_METER.format(bw)).update(train_top1.detach())
        getattr(self, TRAIN_TOP5_BW_METER.format(bw)).update(train_top5.detach())
        return train_loss, output

    def training_step(self, batch, batch_idx):
        images, target = batch
        max_bw = max(self.hparams.slimmable.WEIGHT_BIT_WIDTH)
        train_loss, output = self.training_substep(max_bw, images, target, None, self.loss_fn)
        soft_target = F.softmax(output, dim=1)
        self.backward(self.use_amp, train_loss, self.trainer.optimizers[0], is_substep=True)

        log_dict = OrderedDict({
            LOG_STAGE_LOG_KEY: LogStage.TRAIN_BATCH,
            EPOCH_LOG_KEY: self.current_epoch,
            BATCH_IDX_LOG_KEY: batch_idx,
            NUM_BATCHES_LOG_KEY: self.trainer.nb_training_batches,
        })
        for bw in sorted(self.hparams.slimmable.WEIGHT_BIT_WIDTH, reverse=True)[1:]:  # exclude max_bw
            train_loss, _ = self.training_substep(bw, images, target, soft_target.detach(), self.soft_loss)
            self.backward(self.use_amp, train_loss, self.trainer.optimizers[0], is_substep=True)

        for bw in sorted(self.hparams.slimmable.WEIGHT_BIT_WIDTH, reverse=True):
            log_dict[TRAIN_LOSS_BW_METER.format(bw)] = getattr(self, TRAIN_LOSS_BW_METER.format(bw))
            log_dict[TRAIN_TOP1_BW_METER.format(bw)] = getattr(self, TRAIN_TOP1_BW_METER.format(bw))
            log_dict[TRAIN_TOP5_BW_METER.format(bw)] = getattr(self, TRAIN_TOP5_BW_METER.format(bw))

        output_dict = OrderedDict({
            'loss'.format(max_bw): train_loss,
            'log': log_dict
        })
        return output_dict

    def on_epoch_start(self):
        for bw in self.hparams.slimmable.WEIGHT_BIT_WIDTH:
            getattr(self, TRAIN_LOSS_BW_METER.format(bw)).reset()
            getattr(self, TRAIN_TOP1_BW_METER.format(bw)).reset()
            getattr(self, TRAIN_TOP5_BW_METER.format(bw)).reset()
            getattr(self, VAL_LOSS_BW_METER.format(bw)).reset()
            getattr(self, VAL_TOP1_BW_METER.format(bw)).reset()
            getattr(self, VAL_TOP1_BW_METER.format(bw)).reset()

    def validation_substep(self, bw, batch):
        self.set_bw(bw)
        images, target = batch
        output = self.model(images)
        val_loss, _ = self.multisample_loss(output, target, self.loss_fn)
        val_top1, val_top5 = topk_accuracy(output, target, topk=(1, 5))
        getattr(self, VAL_LOSS_BW_METER.format(bw)).update(val_loss.detach(), images.size(0))
        getattr(self, VAL_TOP1_BW_METER.format(bw)).update(val_top1.detach(), images.size(0))
        getattr(self, VAL_TOP5_BW_METER.format(bw)).update(val_top5.detach(), images.size(0))
        return val_loss

    def validation_step(self, batch, batch_idx):
        log_dict = {LOG_STAGE_LOG_KEY: LogStage.VAL_BATCH,
                    EPOCH_LOG_KEY: self.current_epoch,
                    BATCH_IDX_LOG_KEY: batch_idx,
                    NUM_BATCHES_LOG_KEY: self.trainer.nb_val_batches}
        output_dict = OrderedDict({})
        for bw in sorted(self.hparams.slimmable.WEIGHT_BIT_WIDTH, reverse=True):
            val_loss = self.validation_substep(bw, batch)
            log_dict[VAL_LOSS_BW_METER.format(bw)] = getattr(self, VAL_LOSS_BW_METER.format(bw))
            log_dict[VAL_TOP1_BW_METER.format(bw)] = getattr(self, VAL_TOP1_BW_METER.format(bw))
            log_dict[VAL_TOP5_BW_METER.format(bw)] = getattr(self, VAL_TOP5_BW_METER.format(bw))
            output_dict['loss'.format(bw)] = val_loss
        self.logger.log_metrics(log_dict)
        return output_dict

    def validation_end(self, outputs):
        log_dict = {LOG_STAGE_LOG_KEY: LogStage.EPOCH,
                    EPOCH_LOG_KEY: self.current_epoch}
        for bw in sorted(self.hparams.slimmable.WEIGHT_BIT_WIDTH, reverse=True):
            log_dict[VAL_LOSS_BW_METER.format(bw)] = getattr(self, VAL_LOSS_BW_METER.format(bw))
            log_dict[VAL_TOP1_BW_METER.format(bw)] = getattr(self, VAL_TOP1_BW_METER.format(bw))
            log_dict[VAL_TOP5_BW_METER.format(bw)] = getattr(self, VAL_TOP5_BW_METER.format(bw))
            log_dict[TRAIN_LOSS_BW_METER.format(bw)] = getattr(self, TRAIN_LOSS_BW_METER.format(bw))
            log_dict[TRAIN_TOP1_BW_METER.format(bw)] = getattr(self, TRAIN_TOP1_BW_METER.format(bw))
            log_dict[TRAIN_TOP5_BW_METER.format(bw)] = getattr(self, TRAIN_TOP5_BW_METER.format(bw))
        min_bw = min(self.hparams.slimmable.WEIGHT_BIT_WIDTH)  # use lowerst bw as driving factor for checkpoint
        result = {'log': log_dict,
                  'val_top1': log_dict[VAL_TOP1_BW_METER.format(min_bw)].avg,
                  'val_loss': log_dict[VAL_LOSS_BW_METER.format(min_bw)].avg}
        return result