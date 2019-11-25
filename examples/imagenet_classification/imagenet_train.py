import logging
import os
from apex import amp

import hydra
import torch
import torch.distributed as dist
from imagenet_classification import QuantImageNetClassification
from imagenet_classification.hydra_logger import HydraTestTubeLogger
from pytorch_lightning import Trainer


class CustomDdpTrainer(Trainer):

    def fit(self, model):
        if self.use_ddp:
            assert self.num_gpus == 1, 'Only 1 GPU per process supported'
            rank = self.root_gpu
            self.ddp_train(rank, model)

        elif self.single_gpu:
            self.single_gpu_train(model)

        else:  # On CPU
            # run through amp wrapper
            if self.use_amp:
                raise Exception('amp + cpu is not supported. Please use a GPU option')

            self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())
            self.run_pretrain_routine(model)

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return 1

    def set_nvidia_flags(self, is_slurm_managing_tasks, data_parallel_device_ids):
        if data_parallel_device_ids is None:
            return
        if self.use_ddp:
            logging.info(f'VISIBLE GPUS: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    def init_ddp_connection(self, proc_rank, world_size):
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    def set_distributed_mode(self, distributed_backend, nb_gpu_nodes):
        # skip for CPU
        if self.num_gpus == 0:
            return

        self.single_gpu = True
        if distributed_backend is not None:
            self.use_dp = distributed_backend == 'dp'
            self.use_ddp = distributed_backend == 'ddp'
            self.use_ddp2 = distributed_backend == 'ddp2'

        logging.info

    def ddp_train(self, gpu_nb, model):

        # Flags set by multiproc.py
        self.node_rank = os.environ['NODE_RANK']
        self.proc_rank = os.environ['GLOBAL_RANK']
        self.world_size = os.environ['WORLD_SIZE']

        # show progressbar only on progress_rank 0
        self.show_progress_bar = self.show_progress_bar and self.node_rank == 0 and gpu_nb == 0

        # let the exp know the rank to avoid overwriting logs
        if self.logger is not None:
            self.logger.rank = self.proc_rank

        model.trainer = self
        model.init_ddp_connection(self.proc_rank, self.world_size)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        # MODEL
        # copy model to each gpu
        torch.cuda.set_device(gpu_nb)
        model.cuda(gpu_nb)

        # set model properties before going into wrapper
        self.copy_trainer_model_properties(model)

        # AMP
        # run through amp wrapper before going to distributed DP
        if self.use_amp:
            # An example
            model, optimizers = model.configure_apex(amp, model, self.optimizers, self.amp_level)
            self.optimizers = optimizers

        # Configure ddp
        device_ids = [gpu_nb]
        model = model.configure_ddp(model, device_ids)

        # continue training routine
        self.run_pretrain_routine(model)

    def training_forward(self, batch, batch_nb, opt_idx, hiddens):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_nb:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_nb]
        if len(self.optimizers) > 1:
            args.append(opt_idx)

        # pass hiddens if using tbptt
        if self.truncated_bptt_steps is not None:
            args.append(hiddens)

        # distributed forward
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            output = self.model(*args)

        # single GPU forward
        elif self.single_gpu:
            gpu_id = 0
            if type(self.data_parallel_device_ids) is list:
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        # CPU forward
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_end
        if self.is_overriden('training_end'):
            model_ref = self.get_model()
            output = model_ref.training_end(output)

        # format and reduce outputs accordingly
        output = self.process_output(output, train=True)

        return output


@hydra.main(config_path='conf/train_config.yaml', strict=False)
def main(hparams):
    logging.info(hparams.pretty())
    torch.backends.cudnn.benchmark = True

    model = QuantImageNetClassification(hparams)

    if hparams.IS_DISTRIBUTED:
        distributed_backend = 'ddp'
    else:
        distributed_backend = None

    trainer = CustomDdpTrainer(gpus=str(hparams.GPU),
                               show_progress_bar=False,
                               distributed_backend=distributed_backend,
                               nb_gpu_nodes=hparams.NUM_NODES,
                               row_log_interval=hparams.log.INTERVAL,
                               log_save_interval=hparams.log.SAVE_INTERVAL,
                               weights_summary='top',
                               logger=HydraTestTubeLogger(save_dir=os.getcwd()),
                               use_amp=hparams.MIXED_PRECISION)

    # Call trainer
    trainer.fit(model)


if __name__ == '__main__':
    main()