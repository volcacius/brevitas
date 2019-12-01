import logging
from enum import auto

from logging.config import ConvertingList, ConvertingDict, valid_ident
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from atexit import register

import flatdict
from omegaconf import Config
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.logging.base import rank_zero_only

from .utils import filter_keys, AutoName, IGNORE_VALUE, MissingOptionalDependency

try:
    from trains import Task
except Exception as e:
    Task = MissingOptionalDependency(e)

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

QUEUE_SIZE = 1000

class LogStage(AutoName):
    TRAIN_BATCH = auto()
    VAL_BATCH = auto()
    EPOCH = auto()


class TrainsHydraTestTubeLogger(TestTubeLogger):

    def __init__(
            self,
            save_dir,
            trains_project_name,
            trains_task_name,
            trains_logging_enabled,
            test_tube_logging_enabled,
            name="default",
            description=None,
            debug=False,
            version=None,
            create_git_tag=False):
        super(TrainsHydraTestTubeLogger, self).__init__(
            save_dir,
            name,
            description,
            debug,
            version,
            create_git_tag)
        self.trains_logging_enabled = trains_logging_enabled
        self.test_tube_logging_enabled = test_tube_logging_enabled
        if trains_logging_enabled:
            Task.init(project_name=trains_project_name,
                      task_name=trains_task_name,
                      auto_connect_arg_parser=False)


    @rank_zero_only
    def log_hyperparams(self, hparams):
        self.experiment.debug = self.debug

        if self.test_tube_logging_enabled:
            # Get the hparams as a flattened dict
            hparams_dict = Config.to_container(hparams, resolve=True)
            pseudo_flatten_hparams = flatdict.FlatDict(hparams_dict, delimiter='_')
            flatten_hparams = {}
            for k in pseudo_flatten_hparams.keys():
                val = pseudo_flatten_hparams[k]
                val = str(val) if isinstance(val, list) else val
                val = 'None' if val is None else val
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
        if self.trains_logging_enabled:
            Task.current_task().connect(flatten_hparams)

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
            else:  # EPOCH
                # Add .avg to both cli and tt during epoch logging
                metrics_for_cli[metric + AVG_SUFFIX] = metrics_for_tt[metric + AVG_SUFFIX] = meters[k].avg.item()

        # Log to test-tube
        if self.test_tube_logging_enabled:
            self.experiment.log(metrics_for_tt, global_step=step_num)

        # Log to cli
        if log_stage == LogStage.TRAIN_BATCH or log_stage == LogStage.VAL_BATCH:
            self.log_batch_metrics_cli(log_stage, metrics_for_cli, epoch, batch_idx, num_batches)
        else:  # EPOCH
            self.log_epoch_metrics_cli(metrics_for_cli, epoch)

    @rank_zero_only
    def log_batch_metrics_cli(self, log_stage, meters, epoch, batch_idx, num_batches):
        msg = '[{}][{}/{}][{}]\t'.format(epoch, batch_idx, num_batches, log_stage)
        for k in meters.keys():
            v = meters[k]
            if AVG_SUFFIX not in k and v != IGNORE_VALUE:
                msg += ' {}: {:.4f}'.format(k, v)
                k_avg = k + '_avg'
                if k_avg in meters:
                    msg = msg + ' [{:.4f}]'.format(meters[k_avg])
                msg += '\t'
        self.info(msg)

    @rank_zero_only
    def log_epoch_metrics_cli(self, meters, epoch):
        msg = '[{}][{}]\t'.format(epoch, LogStage.EPOCH)
        for k in meters.keys():
            v = meters[k]
            if v != IGNORE_VALUE:
                msg += ' {}: {:.4f}'.format(k, meters[k])
                msg += '\t'
        self.info(msg)

    @rank_zero_only
    def info(self, msg):
        logging.info(msg)



# From: https://medium.com/@rob.blackbourn/how-to-use-python-logging-queuehandler-with-dictconfig-1e8b1284e27a

def _resolve_handlers(l):
    if not isinstance(l, ConvertingList):
        return l

    # Indexing the list performs the evaluation.
    return [l[i] for i in range(len(l))]


class QueueListenerHandler(QueueHandler):

    def __init__(self, handlers, respect_handler_level=False, auto_run=True, queue=Queue(QUEUE_SIZE)):
        super().__init__(queue)
        handlers = _resolve_handlers(handlers)
        self._listener = QueueListener(
            self.queue,
            *handlers,
            respect_handler_level=respect_handler_level)
        if auto_run:
            self.start()
            register(self.stop)


    def start(self):
        self._listener.start()


    def stop(self):
        if self._listener._thread is not None:
            self._listener.stop()


    def emit(self, record):
        return super().emit(record)
