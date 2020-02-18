import logging
import os

import torch
import torch.distributed as dist
from pytorch_lightning import Trainer


class CustomDdpTrainer(Trainer):

    def resume_optim(self, checkpoint):
        if os.path.exists(checkpoint) and checkpoint.lower().endswith('.ckpt'):
            optimizer_states = checkpoint['optimizer_states']
            for optimizer, opt_state in zip(self.optimizers, optimizer_states):
                optimizer.load_state_dict(opt_state)
                # possibly move to GPU
                if self.root_gpu is not None:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.root_gpu)
            logging.info('Loaded optimizers state at: {}'.format(checkpoint))
        else:
            raise Exception("Can't resume optimizers from checkpoint at {}".format(checkpoint))

    def resume_training_progress(self, checkpoint):
        if os.path.exists(checkpoint) and checkpoint.lower().endswith('.ckpt'):
            self.global_step = checkpoint['global_step']
            self.current_epoch = checkpoint['epoch']
            scheduler_states = checkpoint['lr_schedulers']
            for scheduler, scheduler_state in zip(self.lr_schedulers, scheduler_states):
                scheduler.load_state_dict(scheduler_states)
            logging.info('Loaded training progress at: {}'.format(checkpoint))
        else:
            raise Exception("Can't resume training progress at {}".format(checkpoint))

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

    def ddp_train(self, gpu_nb, model):

        # Flags set by multiproc.py
        self.node_rank = int(os.environ['NODE_RANK'])
        self.proc_rank = int(os.environ['GLOBAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])

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

    def transfer_batch_to_gpu(self, batch, gpu_id, non_blocking):
        # base case: object can be directly moved using `cuda` or `to`
        if callable(getattr(batch, 'cuda', None)):
            return batch.cuda(gpu_id, non_blocking=non_blocking)

        elif callable(getattr(batch, 'to', None)):
            return batch.to(torch.device('cuda', gpu_id), non_blocking=non_blocking)

        # when list
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id, non_blocking)
            return batch

        # when tuple
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id, non_blocking)
            return tuple(batch)

        # when dict
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id, non_blocking)

            return batch

        # nothing matches, return the value as is without transform
        return batch

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if test and len(self.get_test_dataloaders()) > 1:
            args.append(dataloader_idx)

        elif not test and len(self.get_val_dataloaders()) > 1:
            args.append(dataloader_idx)

        # single GPU
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if type(self.data_parallel_device_ids) is list:
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu, non_blocking=True)
            args[0] = batch

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # On CPU or single unwrapped GPU
        if test:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output

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

        # single GPU
        if self.single_gpu:
            gpu_id = 0
            if type(self.data_parallel_device_ids) is list:
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, gpu_id, non_blocking=True)
            args[0] = batch

        # distributed forward
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            output = self.model(*args)
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_end
        if self.is_overriden('training_end'):
            model_ref = self.get_model()
            output = model_ref.training_end(output)

        # format and reduce outputs accordingly
        output = self.process_output(output, train=True)

        return output
