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


class TensorNorm(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.01):
        super(TensorNorm, self).__init__()
        self.register_buffer('running_mean', torch.tensor(0))
        self.register_buffer('running_var', torch.tensor(1))
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input):
        if self.training:
            mean = input.mean()
            unbias_var = input.var(unbiased=True)
            self.running_mean = (1-self.momentum) * self.running_mean + mean.detach() * self.momentum
            self.running_var = (1-self.momentum) * self.running_var + unbias_var.detach() * self.momentum
            biased_var = input.var(unbiased=False)
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            output = (input - mean) * inv_std * self.weight + self.bias
            return output
        else:
            return (input - self.running_mean) * (1.0 / (self.running_var+self.eps).pow(0.5)) * self.weight + self.bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        running_mean_key = prefix + 'running_mean'
        running_var_key = prefix + 'running_var'

        if running_mean_key in state_dict and running_var_key in state_dict:
            state_dict[running_mean_key] = state_dict[running_mean_key].mean()
            state_dict[running_var_key] = state_dict[running_var_key].mean()
        if weight_key in state_dict and bias_key in state_dict:
            state_dict[bias_key] = state_dict[bias_key].mean()
            state_dict[weight_key] = state_dict[weight_key].mean()
        super(TensorNorm, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                                       missing_keys, unexpected_keys, error_msgs)



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
            for k in list(state_dict.keys()):
                if k.startswith(bn_prefix):
                    del state_dict[k]
