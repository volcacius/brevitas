import logging
import os

import numpy as np
from pytorch_lightning.callbacks.pt_callbacks import Callback

"""
Based on PL ModelCheckpoint
"""
class BestModelCheckpoint(Callback):

    def __init__(self, filepath, monitor, mode):
        super(BestModelCheckpoint, self).__init__()

        self.monitor = monitor
        self.filepath = os.path.abspath(filepath)
        self.save_weights_only = True

        if mode not in ['min', 'max']:
            raise Exception("Mode {} not recognized".format(mode))

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def save_model(self, filepath):
        # make paths
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # delegate the saving to the model
        self.save_function(filepath)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            raise Exception("Can't save best model without the required metric being logged")
        else:
            if self.monitor_op(current, self.best):
                    self.best = current
                    self.save_model(self.filepath)
                    logging.info(f'\nEpoch {epoch + 1:05d}: {self.monitor} improved'
                                 f' from {self.best:0.5f} to {current:0.5f},'
                                 f' saving model to {self.filepath}')

