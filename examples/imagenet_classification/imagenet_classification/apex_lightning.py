
# All rights reserved. Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met: 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 2. Redistributions in binary form must reproduce the
# above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution. 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this software without specific prior written
# permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
from pytorch_lightning.pt_overrides.override_data_parallel import LightningDistributedDataParallel


class LightningApexDistributedDataParallel(ApexDistributedDataParallel,
                                           LightningDistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    Inherit also from LightningDistributedDataParallel as the second parent so that
    isistance(obj, LightningDistributedDataParallel) = True, which is leveraged accross pytorch-lightning
    Modified from:
    https://github.com/NVIDIA/apex/blob/47da14a095e87bcd5b1ba176ddb93dd71521b9b7/apex/parallel/distributed.py
    """
    def forward(self, *inputs, **kwargs):  # pragma: no cover
        # --------------
        # LIGHTNING MOD
        # --------------
        # normal
        # output = self.module(*inputs, **kwargs)
        # lightning
        if self.module.training:
            result = self.module.training_step(*inputs, **kwargs)
        elif self.module.testing:
            result = self.module.test_step(*inputs, **kwargs)
        else:
            result = self.module.validation_step(*inputs, **kwargs)

        if self.prof:
            torch.cuda.nvtx.range_push("forward pass DDP logic")

        if not self._disable_allreduce:
            if not self.delay_allreduce:
                param_list = [param for param in self.module.parameters() if param.requires_grad]

                # Conditions under which to refresh self.record
                # Forward has the authority to set needs_refresh to True, but only allreduce_params
                # in backward has the authority to set needs_refresh to False.
                # Parentheses are not necessary for correct order of operations, but make the intent clearer.
                if ((not self.active_params) or
                        (len(param_list) != len(self.active_params)) or
                        any([param1 is not param2 for param1, param2 in zip(param_list, self.active_params)])):
                    self.needs_refresh = True

                if self.needs_refresh:
                    self.active_i_buckets = []
                    self.buckets = []
                    self.tmp_buckets = [[], [], []]  # [running half, float, double buckets]
                    self.tmp_numels = [0, 0, 0]
                    self.bucket_sizes = []
                    self.param_id_to_active_i = {id(param): i for i, param in enumerate(param_list)}
                    self.param_id_to_bucket = {}
                    self.bucket_pgs = []
                    self.bucket_streams = []
                    self.bucket_events = []
                else:
                    # self.buckets = [[None for _ in range(self.bucket_sizes[i])]
                    #                 for i in range(self.num_buckets)]
                    if not self.buckets:
                        self.buckets = [[None for _ in range(self.bucket_sizes[i])]
                                        for i in range(self.num_buckets)]
                    else:
                        assert len(self.buckets) == self.num_buckets, "len(buckets) = {}, expected {}".format(
                            len(self.buckets), self.num_buckets)
                        for b, bucket in enumerate(self.buckets):
                            assert len(bucket) == self.bucket_sizes[b], "len(buckets[{}]) = {}, expected {})".format(
                                b, len(self.buckets[b]), self.bucket_sizes[b])
                            for i in range(len(bucket)):
                                bucket[i] = None

                    if self.allreduce_communicators:
                        self.bucket_pgs = self.allreduce_communicators[0]
                        self.bucket_streams = self.allreduce_communicators[1]
                        self.bucket_events = [torch.cuda.Event(enable_timing=False,
                                                               blocking=False) for _ in
                                              range(self.num_allreduce_streams)]
                    else:
                        if self.allreduce_different_streams:
                            if not self.bucket_pgs:
                                self.bucket_pgs = [dist.new_group() for _ in range(self.num_allreduce_streams)]
                                for i, bg in enumerate(self.bucket_pgs):
                                    print("rank {} created group {} with backend {}".format(
                                        dist.get_rank(), i, dist.get_backend(bg)))
                        if self.allreduce_different_streams:
                            if not self.bucket_streams:
                                self.bucket_streams = [torch.cuda.Stream() for _ in range(self.num_allreduce_streams)]
                                self.bucket_events = [torch.cuda.Event(enable_timing=False,
                                                                       blocking=False) for _ in
                                                      range(self.num_allreduce_streams)]
                        else:
                            if not self.bucket_streams:
                                self.bucket_streams = [torch.cuda.Stream()]
                                self.bucket_events = [torch.cuda.Event(enable_timing=False, blocking=False)]

                    self.buckets_ready_size = [0 for i in range(self.num_buckets)]
                    if (self.retain_allreduce_buffers):
                        self.allreduce_buffers = [None for _ in range(self.num_buckets)]
                    self.next_bucket = 0
                    self.ready_buckets_not_reduced = set()

                self.active_params = param_list

            self.callback_queued = False

        if self.prof:
            torch.cuda.nvtx.range_pop()

        return result