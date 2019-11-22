from enum import Enum, auto
import logging
from trains import Task


import flatdict
from omegaconf import Config
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.logging.base import rank_zero_only
from .utils import filter_keys

LOG_STAGE_LOG_KEY = 'log_stage'
BATCH_IDX_LOG_KEY = 'batch_idx'
NUM_BATCHES_LOG_KEY = 'num_batches'
EPOCH_LOG_KEY = 'epoch'

LOSS_LOG_KEY = 'loss'
TOP1_LOG_KEY = 'top1'
TOP5_LOG_KEY = 'top5'

TRAIN_ = 'train_'
VAL_ = 'val_'
METER_SUFFIX = '_meter'
AVG_SUFFIX = '_avg'

TRAIN_LOSS_METER = TRAIN_ + LOSS_LOG_KEY + METER_SUFFIX
TRAIN_TOP1_METER = TRAIN_ + TOP1_LOG_KEY + METER_SUFFIX
TRAIN_TOP5_METER = TRAIN_ + TOP5_LOG_KEY + METER_SUFFIX

VAL_LOSS_METER = VAL_ + LOSS_LOG_KEY + METER_SUFFIX
VAL_TOP1_METER = VAL_ + TOP1_LOG_KEY + METER_SUFFIX
VAL_TOP5_METER = VAL_ + TOP5_LOG_KEY + METER_SUFFIX


class AutoName(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
         return name

    def __str__(self):
        return self.value


class LogStage(AutoName):
    TRAIN_BATCH = auto()
    VAL_BATCH = auto()
    EPOCH = auto()


class HydraTestTubeLogger(TestTubeLogger):

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.debug = self.debug

        # Get the hparams as a flattened dict
        params_dict = Config.to_container(params, resolve=True)
        pseudo_flatten_hparams = flatdict.FlatDict(params_dict, delimiter='_')
        flatten_hparams = {}
        for k in pseudo_flatten_hparams.keys():
            val = pseudo_flatten_hparams[k]
            val = str(val) if isinstance(val, list) else val
            flatten_hparams[k.lower()] = val

        # Log into test-tube. Requires to pass an object with a __dict__ attribute
        class TempStruct:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        self.experiment.argparse(TempStruct(**flatten_hparams))

        # Log into tensorboard hparams plugin (supported in Pytorch 1.3)
        if hasattr(self.experiment, 'add_hparams'):
            self.experiment.add_hparams(flatten_hparams, metric_dict={})

        # Log into trains
        Task.current_task().connect(flatten_hparams)
        model_config, _ = filter_keys(flatten_hparams, ['model', 'preprocess'], return_dict=True)
        Task.current_task().set_model_config(model_config)

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        self.experiment.debug = self.debug

        # Extract metadatas, shown only on the cli
        num_batches = metrics.pop(NUM_BATCHES_LOG_KEY, None)
        batch_idx = metrics.pop(BATCH_IDX_LOG_KEY, None)
        epoch = metrics.pop(EPOCH_LOG_KEY)
        log_stage = metrics.pop(LOG_STAGE_LOG_KEY)

        # Extract required values from meters
        meters, others = filter_keys(metrics.items(), [METER_SUFFIX], return_dict=True)
        metrics_for_tt = others.copy()
        metrics_for_cli = others.copy()
        for k in meters.keys():
            metric = k.rstrip(METER_SUFFIX)
            if log_stage == LogStage.TRAIN_BATCH or log_stage == LogStage.VAL_BATCH:
                # Add .val to both cli and tt during batch logging
                metrics_for_cli[metric] = metrics_for_tt[metric] = meters[k].val.item()
                # add .avg only to cli during batch logging
                metrics_for_cli[metric + AVG_SUFFIX] = meters[k].avg.item()
            else:
                # Add .avg to both cli and tt during epoch logging
                metrics_for_cli[metric + AVG_SUFFIX] = metrics_for_tt[metric + AVG_SUFFIX] = meters[k].avg.item()

        # Log to test-tube
        self.experiment.log(metrics_for_tt, global_step=step_num)
        # Log to cli
        self.log_metrics_cli(log_stage, metrics_for_cli, epoch, batch_idx, num_batches)

    @rank_zero_only
    def log_metrics_cli(self, log_stage, meters, epoch, batch_idx, num_batches):
        if batch_idx is not None and num_batches is not None:
            msg = '[{}][{}/{}][{}]\t'.format(epoch, batch_idx, num_batches, log_stage)
        else:
            msg = '[{}][{}]\t'.format(epoch, log_stage)
        for k in meters.keys():
            if AVG_SUFFIX not in k:
                msg += ' {}: {:.4f}'.format(k, meters[k])
                k_avg = k + '_avg'
                if k_avg in meters:
                    msg = msg + ' [{:.4f}]'.format(meters[k_avg])
                msg += '\t'
        self.info(msg)

    @rank_zero_only
    def info(self, msg):
        logging.info(msg)
