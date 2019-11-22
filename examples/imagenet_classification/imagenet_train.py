import os
import logging

import hydra
import torch

from pytorch_lightning import Trainer

from imagenet_classification import QuantImageNetClassification
from imagenet_classification.hydra_logger import HydraTestTubeLogger



@hydra.main(config_path='conf/train_config.yaml', strict=False)
def main(hparams):
    logging.info(hparams.pretty())
    torch.backends.cudnn.benchmark = True

    model = QuantImageNetClassification(hparams)

    if hparams.IS_DISTRIBUTED:
        distributed_backend = 'ddp'
    else:
        distributed_backend = None

    trainer = Trainer(gpus=hparams.GPUS,
                      show_progress_bar=False,
                      distributed_backend=distributed_backend,
                      row_log_interval=hparams.log.INTERVAL,
                      log_save_interval=hparams.log.SAVE_INTERVAL,
                      weights_summary='top',
                      logger=HydraTestTubeLogger(save_dir=os.getcwd()),
                      use_amp=hparams.MIXED_PRECISION)

    # Call trainer
    trainer.fit(model)


if __name__ == '__main__':
    main()