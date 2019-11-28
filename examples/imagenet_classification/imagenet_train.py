import logging
import os
from pathlib import Path

import hydra
import torch
from imagenet_classification import QuantImageNetClassification
from imagenet_classification.hydra_logger import HydraTestTubeLogger
from imagenet_classification.pl_overrides.pl_callbacks import BestModelCheckpoint
from imagenet_classification.pl_overrides.pl_trainer import CustomDdpTrainer
from trains import Task


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

    logging.info(hparams.pretty())
    exp_timestamp = Path(os.getcwd()).parents[0].parts[-1]
    task_name = '{}_{}'.format(hparams.NAME_PREFIX, exp_timestamp)
    if hparams.log.TRAINS_LOGGING:
        Task.init(project_name=hparams.model.ARCH,
                  task_name=task_name,
                  auto_connect_arg_parser=False)

    ckpt_callback = BestModelCheckpoint(filepath=os.path.join(os.getcwd(), task_name + '.ckpt'),
                                        monitor='val_top1',
                                        mode='max')

    # A single GPU id has to be passed as a string, otherwise it will be interpreted as # of GPUs
    trainer = CustomDdpTrainer(gpus=str(hparams.GPU),
                               max_nb_epochs=hparams.EPOCHS,
                               show_progress_bar=False,
                               distributed_backend=distributed_backend,
                               nb_gpu_nodes=hparams.NUM_NODES,
                               row_log_interval=hparams.log.INTERVAL,
                               log_save_interval=hparams.log.SAVE_INTERVAL,
                               weights_summary='top',
                               checkpoint_callback=ckpt_callback,
                               logger=HydraTestTubeLogger(save_dir=os.getcwd()),
                               use_amp=hparams.MIXED_PRECISION)

    # Call trainer
    trainer.fit(model)


if __name__ == '__main__':
    main()