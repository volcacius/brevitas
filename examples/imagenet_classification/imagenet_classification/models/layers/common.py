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

import torch
import torch.nn.functional as F
import torch.nn as nn

from brevitas.quant_tensor import pack_quant_tensor
from brevitas.nn.quant_bn import mul_add_from_bn

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
        missing_bias_keys = None
        if self.merge_bn:
            _merge_bn_layers(
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

def _merge_bn_layers(conv_bn_tuples, bn_eps, prefix, state_dict):
    for conv_mod, conv_name, bn_name, in conv_bn_tuples:
        bn_prefix = prefix + bn_name
        bn_weight_key = '.'.join([bn_prefix, 'weight'])
        bn_bias_key = '.'.join([bn_prefix,'bias'])
        bn_mean_key = '.'.join([bn_prefix, 'running_mean'])
        bn_var_key = '.'.join([bn_prefix, 'running_var'])
        bn_keys = [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]
        if any(i in state_dict for i in bn_keys):
            mul_factor, add_factor = mul_add_from_bn(
                bn_mean=state_dict[bn_mean_key],
                bn_var=state_dict[bn_var_key],
                bn_eps=bn_eps,
                bn_weight=state_dict[bn_weight_key],
                bn_bias=state_dict[bn_bias_key],
                affine_only=False)
            mul_shape = conv_mod.per_output_channel_broadcastable_shape
            conv_mod.weight.data *= mul_factor.view(mul_shape)

            conv_bias_key = prefix + conv_name + '.bias'
            if conv_mod.bias is not None:
                 assert conv_bias_key in state_dict
                 state_dict[conv_bias_key] += add_factor
            else:
                conv_mod.bias = nn.Parameter(add_factor)
                # add it to the dict any to avoid missing key error
                state_dict[conv_bias_key] = add_factor

            # Get rid of statistics after using them
            for k in list(state_dict.keys()):
                if k.startswith(bn_prefix):
                    del state_dict[k]
