from functools import partial

import torch
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from torch import nn

from ...utils import lowercase_keys
from .common import MergeBn

__all__ = ['MakeLayerWithDefaults']


def make_layer_with_defaults(layer, kwargs_list):
    cumulative_kwargs = {}
    for kwargs in kwargs_list:
        cumulative_kwargs.update(kwargs)
    return partial(layer, **lowercase_keys(cumulative_kwargs))


class MakeLayerWithDefaults:

    def __init__(self, params):
        self.make_quant_conv2d = make_layer_with_defaults(
            make_quant_conv2d, [params.quant_weights, params.conv])
        self.make_quant_linear = make_layer_with_defaults(
            make_quant_linear, [params.quant_weights, params.linear])
        self.make_quant_relu = make_layer_with_defaults(
            make_quant_relu, [params.quant_act, params.quant_relu])
        self.make_quant_hard_tanh = make_layer_with_defaults(
            make_quant_hard_tanh, [params.quant_act, params.quant_hard_tanh])
        self.make_quant_avg_pool = make_layer_with_defaults(
            make_quant_avg_pool, [params.quant_avg_pool])
        self.make_hadamard_classifier = make_layer_with_defaults(
            make_hadamard_classifier, [])


# File   : batchnorm_reimpl.py
# Author : acgtyrant
# Date   : 11/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

class MeanOnlyBatchNorm2d(torch.jit.ScriptModule):
    __constants__ = ['momentum']

    def __init__(self, features, momentum):
        super(MeanOnlyBatchNorm2d, self).__init__()
        self.momentum = momentum
        self.bias = nn.Parameter(torch.empty(features).fill_(0.0))
        self.register_buffer('running_mean', torch.empty(features).fill_(0.0))

    @torch.jit.script_method
    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        permuted_input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, -1)
        if self.training:
            mean = permuted_input_.mean(dim=1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * (mean.detach())
            output = (input_ - mean.view(1, -1, 1, 1)) + self.bias.view(1, -1, 1, 1)
        else:
            output = (input_ - self.running_mean.view(1, -1, 1, 1)) + self.bias.view(1, -1, 1, 1)
        return output


def make_bn(merge_bn, features, eps, momentum):
    if merge_bn is None:
        return nn.BatchNorm2d(features, eps=eps, momentum=momentum)
    elif merge_bn == MergeBn.ALL_TO_IDENTITY or merge_bn == MergeBn.IDENTITY:
        return nn.Identity()
    elif merge_bn == MergeBn.ALL_TO_MEAN_ONLY or merge_bn == MergeBn.MEAN_ONLY:
        return MeanOnlyBatchNorm2d(features, momentum)
    else:
        raise Exception("Merge BN strategy not recognized: {}".format(merge_bn))


def make_quant_conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      groups,
                      bias,
                      bit_width,
                      padding,
                      padding_type,
                      enable_bias_quant,
                      weight_quant_type,
                      weight_norm_impl_type,
                      weight_scaling_impl_type,
                      weight_scaling_stats_op,
                      weight_scaling_per_output_channel,
                      weight_restrict_scaling_type,
                      weight_narrow_range,
                      weight_scaling_min_val):
    bias_quant_type = QuantType.INT if enable_bias_quant else QuantType.FP
    return qnn.QuantConv2d(in_channels,
                           out_channels,
                           groups=groups,
                           kernel_size=kernel_size,
                           padding=padding,
                           stride=stride,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=enable_bias_quant,
                           compute_output_scale=enable_bias_quant,
                           padding_type=padding_type,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_norm_impl_type=weight_norm_impl_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val)


def make_quant_linear(in_channels,
                      out_channels,
                      bias,
                      bit_width,
                      enable_bias_quant,
                      weight_quant_type,
                      weight_norm_impl_type,
                      weight_scaling_impl_type,
                      weight_scaling_stats_op,
                      weight_scaling_per_output_channel,
                      weight_restrict_scaling_type,
                      weight_narrow_range,
                      weight_scaling_min_val):
    bias_quant_type = QuantType.INT if enable_bias_quant else QuantType.FP
    return qnn.QuantLinear(in_channels, out_channels,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=enable_bias_quant,
                           compute_output_scale=enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_norm_impl_type=weight_norm_impl_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val)


def make_quant_relu(bit_width,
                    quant_type,
                    norm_impl_type,
                    scaling_impl_type,
                    scaling_per_channel,
                    restrict_scaling_type,
                    scaling_stats_op,
                    scaling_min_val,
                    max_val,
                    return_quant_tensor,
                    per_channel_broadcastable_shape,
                    norm_restats,
                    scaling_restats):
    return qnn.QuantReLU(bit_width=bit_width,
                         quant_type=quant_type,
                         norm_impl_type=norm_impl_type,
                         scaling_impl_type=scaling_impl_type,
                         scaling_per_channel=scaling_per_channel,
                         restrict_scaling_type=restrict_scaling_type,
                         scaling_min_val=scaling_min_val,
                         scaling_stats_op=scaling_stats_op,
                         max_val=max_val,
                         return_quant_tensor=return_quant_tensor,
                         per_channel_broadcastable_shape=per_channel_broadcastable_shape,
                         norm_restats=norm_restats,
                         scaling_restats=scaling_restats)


def make_quant_hard_tanh(bit_width,
                         quant_type,
                         norm_impl_type,
                         scaling_impl_type,
                         scaling_per_channel,
                         restrict_scaling_type,
                         scaling_stats_op,
                         scaling_min_val,
                         threshold,
                         return_quant_tensor,
                         per_channel_broadcastable_shape,
                         norm_restats,
                         scaling_restats):
    return qnn.QuantHardTanh(bit_width=bit_width,
                             quant_type=quant_type,
                             scaling_per_channel=scaling_per_channel,
                             norm_impl_type=norm_impl_type,
                             scaling_impl_type=scaling_impl_type,
                             restrict_scaling_type=restrict_scaling_type,
                             scaling_min_val=scaling_min_val,
                             scaling_stats_op=scaling_stats_op,
                             max_val=threshold,
                             min_val=-threshold,
                             per_channel_broadcastable_shape=per_channel_broadcastable_shape,
                             return_quant_tensor=return_quant_tensor,
                             norm_restats=norm_restats,
                             scaling_restats=scaling_restats)


def make_quant_avg_pool(bit_width,
                        kernel_size,
                        stride,
                        signed,
                        quant_type):
    return qnn.QuantAvgPool2d(kernel_size=kernel_size,
                              quant_type=quant_type,
                              signed=signed,
                              stride=stride,
                              min_overall_bit_width=bit_width,
                              max_overall_bit_width=bit_width)


def make_hadamard_classifier(in_channels,
                             out_channels,
                             fixed_scale):
    return qnn.HadamardClassifier(in_channels=in_channels,
                                  out_channels=out_channels,
                                  fixed_scale=fixed_scale)