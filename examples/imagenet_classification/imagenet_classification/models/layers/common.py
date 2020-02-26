""" drop_connect
Copyright 2019 Ross Wightman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABCMeta, abstractmethod
from enum import auto

import torch
import torch.nn.functional as F
import torch.nn as nn

from brevitas.quant_tensor import pack_quant_tensor
from brevitas.nn.quant_bn import mul_add_from_bn

from brevitas.utils.python_utils import AutoName


class MergeBn(AutoName):
    ALL_TO_IDENTITY = auto()
    ALL_REINIT_PER_CHANNEL = auto()
    ALL_REINIT_PER_TENSOR = auto()
    ALL_REINIT_PER_TENSOR_AVE = auto()
    RESET_STATS = auto()
    STATS_ONLY = auto()
    LOG_BN = auto()


def drop_connect(inputs, training: bool = False, drop_connect_rate: float = 0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


def multisample_dropout_classify(x, classifier, samples, rate, training):
    x, scale, bit_width = x
    x = x.view(x.size(0), -1)
    if training and samples == 1:
        out = F.dropout(x, p=rate)
        out = classifier(pack_quant_tensor(out, scale, bit_width))
        return out
    if training and samples > 1:
        out_list = []
        for i in range(samples):
            out = F.dropout(x, p=rate)
            out = classifier(pack_quant_tensor(out, scale, bit_width))
            out_list.append(out)
        return tuple(out_list)
    else:
        out = classifier(pack_quant_tensor(x, scale, bit_width))
        return out


def residual_add_drop_connect(x, other, training, drop_connect_rate):
    tensor, scale, bit_width = x
    if drop_connect_rate > 0.:
        tensor = drop_connect(tensor, training, drop_connect_rate)
    if training:
        tensor += other.tensor
        x = pack_quant_tensor(tensor, scale, bit_width + other.bit_width)
        return x
    else:
        x = pack_quant_tensor(tensor, scale, bit_width)
        x += other
        return x


class MergeBnMixin:
    __metaclass__ = ABCMeta

    @abstractmethod
    def conv_bn_tuples(self):
        pass

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs):
        if self.merge_bn is not None:
            _merge_bn_layers(
                self.merge_bn,
                conv_bn_tuples=self.conv_bn_tuples(),
                bn_eps=self.bn_eps,
                prefix=prefix,
                state_dict=state_dict)
        super(MergeBnMixin, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs)


def _merge_bn_layers(merge_bn, conv_bn_tuples, bn_eps, prefix, state_dict):
    for conv_mod, conv_name, bn_name, in conv_bn_tuples:
        bn_prefix = prefix + bn_name
        bn_weight_key = '.'.join([bn_prefix, 'weight'])
        bn_bias_key = '.'.join([bn_prefix,'bias'])
        bn_mean_key = '.'.join([bn_prefix, 'running_mean'])
        bn_var_key = '.'.join([bn_prefix, 'running_var'])
        bn_keys = [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]
        if merge_bn == MergeBn.RESET_STATS:
            state_dict[bn_mean_key].fill_(0.0)
            state_dict[bn_var_key].fill_(1.0)
            return
        if merge_bn == MergeBn.LOG_BN:
            return
        if any(i in state_dict for i in bn_keys):
            mul_factor, add_factor = mul_add_from_bn(
                bn_mean=state_dict[bn_mean_key],
                bn_var=state_dict[bn_var_key],
                bn_eps=bn_eps,
                bn_weight=state_dict[bn_weight_key],
                bn_bias=state_dict[bn_bias_key],
                stats_only=merge_bn==MergeBn.STATS_ONLY)
            mul_shape = conv_mod.per_output_channel_broadcastable_shape
            conv_weight_key = prefix + conv_name + '.weight'
            conv_bias_key = prefix + conv_name + '.bias'
            state_dict[conv_weight_key] *= mul_factor.view(mul_shape)

            if conv_mod.bias is not None and conv_bias_key in state_dict:
                 state_dict[conv_bias_key] += add_factor
            elif conv_mod.bias is not None and not conv_bias_key in state_dict:
                state_dict[conv_bias_key] = add_factor
            else:
                conv_mod.bias = nn.Parameter(add_factor)
                # add it to the dict any to avoid missing key error
                state_dict[conv_bias_key] = add_factor

            # Get rid of statistics after using them
            if merge_bn == MergeBn.ALL_TO_IDENTITY or \
                    merge_bn == MergeBn.ALL_REINIT_PER_TENSOR or \
                    merge_bn == MergeBn.ALL_REINIT_PER_TENSOR_AVE:
                for k in list(state_dict.keys()):
                    if k.startswith(bn_prefix):
                        del state_dict[k]
                if merge_bn == MergeBn.ALL_REINIT_PER_TENSOR or merge_bn == MergeBn.ALL_REINIT_PER_TENSOR_AVE:
                    state_dict[bn_weight_key] = torch.tensor(1.0)
                    state_dict[bn_bias_key] = torch.tensor(0.0)
                    state_dict[bn_mean_key] = torch.tensor(0.0)
                    state_dict[bn_var_key] = torch.tensor(1.0)
            elif merge_bn == MergeBn.STATS_ONLY:
                state_dict[bn_mean_key].fill_(0.0)
                state_dict[bn_var_key].fill_(1.0)
            elif merge_bn == MergeBn.ALL_REINIT_PER_CHANNEL:
                state_dict[bn_weight_key].fill_(1.0)
                state_dict[bn_bias_key].fill_(0.0)
                state_dict[bn_mean_key].fill_(0.0)
                state_dict[bn_var_key].fill_(1.0)
            else:
                raise Exception("Merge BN strategy not recognized: {}".format(merge_bn))
