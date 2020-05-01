import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from imagenet_classification import QuantImageNetClassification
from imagenet_classification.hydra_logger import TrainsHydraTestTubeLogger, QueueListenerHandler
from imagenet_classification.pl_overrides.pl_callbacks import BestModelCheckpoint
from imagenet_classification.pl_overrides.pl_trainer import CustomDdpTrainer


@hydra.main(config_path='conf/train_config.yaml', strict=True)
def main(hparams):
    torch.backends.cudnn.benchmark = True

    model = QuantImageNetClassification(hparams)

    if hparams.IS_DISTRIBUTED:
        distributed_backend = 'ddp'
        # GPU ids are always normalize to start from 0 thanks to VISIBLE DEVICES
        hparams.log.TRAINS_LOGGING &= hparams.GPU == 0
    else:
        distributed_backend = None

    # Init logger
    exp_timestamp = Path(os.getcwd()).parents[0].parts[-1]
    task_name = '{}_{}'.format(hparams.NAME_PREFIX, exp_timestamp)
    logger = TrainsHydraTestTubeLogger(save_dir=os.getcwd(),
                                       trains_logging_enabled=hparams.log.TRAINS_LOGGING,
                                       test_tube_logging_enabled=hparams.log.TEST_TUBE_LOGGING,
                                       trains_project_name=hparams.model.ARCH,
                                       trains_task_name=task_name)
    logger.info(hparams.pretty())

    # Init checkpoint callback
    ckpt_callback = BestModelCheckpoint(filepath=os.path.join(os.getcwd(), task_name + '.ckpt'),
                                        monitor='val_top1_ema',
                                        mode='max')

    # A single GPU id has to be passed as a string, otherwise it will be interpreted as # of GPUs
    trainer = CustomDdpTrainer(gpus=str(hparams.GPU) if hparams.GPU is not None else hparams.GPU,
                               max_nb_epochs=hparams.EPOCHS,
                               show_progress_bar=False,
                               early_stop_callback=None,
                               distributed_backend=distributed_backend,
                               nb_gpu_nodes=hparams.NUM_NODES,
                               row_log_interval=hparams.log.INTERVAL,
                               log_save_interval=hparams.log.SAVE_INTERVAL,
                               weights_summary='top',
                               checkpoint_callback=ckpt_callback,
                               logger=logger,
                               use_amp=hparams.MIXED_PRECISION)

    # Call trainer
    if hparams.EVALUATE_ONLY:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    try:
        main()
    except:
        for h in logging.getLogger().handlers:
            if isinstance(h, QueueListenerHandler):
                h.stop()
        raise

