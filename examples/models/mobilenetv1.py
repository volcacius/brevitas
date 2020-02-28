"""
Modified from: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models

MIT License

Copyright (c) 2019 Xilinx, Inc (Alessandro Pappalardo)
Copyright (c) 2018 Oleg Sémery

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__all__ = ['quant_mobilenet_v1']

import torch
from torch import nn
from torch.nn import Sequential

from brevitas.core import ZERO_HW_SENTINEL_NAME
from brevitas.quant_tensor import pack_quant_tensor

from .common import make_quant_conv2d, make_quant_linear, make_quant_relu, make_quant_avg_pool
from .export_utils import *


FIRST_LAYER_BIT_WIDTH = 8
EXPORT = True


class DwsConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bit_width,
                 pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(in_channels=in_channels,
                                 out_channels=in_channels,
                                 groups=in_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=stride,
                                 weight_bit_width=bit_width,
                                 act_bit_width=bit_width)
        self.pw_conv = ConvBlock(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 padding=0,
                                 weight_bit_width=bit_width,
                                 act_bit_width=bit_width,
                                 activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def export(self, name_prefix):
        export_list = []
        export_list.extend(self.dw_conv.export(name_prefix + '_dw'))
        export_list.extend(self.pw_conv.export(name_prefix + '_pw'))
        return export_list

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 weight_bit_width,
                 act_bit_width,
                 stride=1,
                 padding=0,
                 groups=1,
                 bn_eps=1e-5,
                 activation_scaling_per_channel=False):
        super(ConvBlock, self).__init__()
        self.conv = make_quant_conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      groups=groups,
                                      bias=False,
                                      bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = make_quant_relu(bit_width=act_bit_width,
                                          per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                          scaling_per_channel=activation_scaling_per_channel,
                                          return_quant_tensor=True)
        self.input_2d_shape = None
        self.output_2d_shape = None
        self.acc_scale = None
        self.acc_bit_width = None
        self.output_bit_width = None

    def export(self, name_prefix):
        factors_tuple = scale_bias_fusion(
            self.bn,
            scale_factor_init=self.acc_scale,
            bias_factor_init=self.conv.bias if self.conv.bias else 0.0)
        acc_scale_factor, acc_bias_factor, weight_sign_factor = factors_tuple
        # Threshold
        threshold = hls_threshold_string(
            self.activation,
            hls_var_name='{}threshold'.format(name_prefix.lower()),
            acc_bit_width=self.acc_bit_width,
            acc_scale_factor=acc_scale_factor,
            acc_bias_factor=acc_bias_factor,
            output_bit_width=self.output_bit_width)
        # Weight
        weight_bit_width_impl = self.conv.weight_quant.tensor_quant.msb_clamp_bit_width_impl
        weight_bit_width = weight_bit_width_impl(getattr(self.conv.weight_quant, ZERO_HW_SENTINEL_NAME))
        weight_bit_width = weight_bit_width.int().item()
        conv_weight = hls_weight_string(
            self.conv,
            weight_bit_width=weight_bit_width,
            hls_var_name='{}weight'.format(name_prefix.lower()),
            sign_factor=weight_sign_factor)
        # Config
        config_list = hls_config_string(
            self.conv,
            name_prefix.upper(),
            weight_bit_width=weight_bit_width,
            input_2d_shape=self.input_2d_shape,
            output_2d_shape=self.output_2d_shape)
        # Return as a list of a single tuple
        export_tuple = conv_weight, threshold, config_list
        return [export_tuple]

    def forward(self, x):
        input_shape = x.tensor.shape
        x, acc_scale, acc_bit_width = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if EXPORT:
            self.input_2d_shape = (input_shape[2], input_shape[3])
            self.acc_scale = acc_scale.detach()
            self.acc_bit_width = acc_bit_width.int().item()
            self.output_bit_width = x.bit_width.int().item()
            self.output_2d_shape = (x.tensor.shape[2], x.tensor.shape[3])
        return x


class MobileNet(nn.Module):

    def __init__(self,
                 channels,
                 first_stage_stride,
                 bit_width,
                 in_channels=3,
                 num_classes=1000):
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]

        self.features = Sequential()
        init_block = ConvBlock(in_channels=in_channels,
                               out_channels=init_block_channels,
                               kernel_size=3,
                               stride=2,
                               weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                               activation_scaling_per_channel=True,
                               act_bit_width=bit_width)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                mod = DwsConvBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   stride=stride,
                                   bit_width=bit_width,
                                   pw_activation_scaling_per_channel=pw_activation_scaling_per_channel)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = make_quant_avg_pool(kernel_size=7,
                                              stride=1,
                                              signed=False,
                                              bit_width=bit_width)
        self.output = make_quant_linear(in_channels, num_classes,
                                        bias=True,
                                        enable_bias_quant=True,
                                        bit_width=bit_width,
                                        weight_scaling_per_output_channel=False)

    def export(self, input_sample):
        self.eval()
        with torch.no_grad():
            self.forward(input_sample)
        export_list = []
        export_list.extend(self.features[0].export(name_prefix='conv0'))
        for i, stage in enumerate(self.features[1:]):
            for j, block in enumerate(stage):
                name_prefix = 'layer{}_block{}'.format(i, j)
                export_list.extend(block.export(name_prefix=name_prefix))
        weight_list, threshold_list, config_list = zip(*export_list)
        return weight_list, threshold_list, config_list

    def forward(self, x):
        x = pack_quant_tensor(x, torch.tensor(0.226).to(x.device), torch.tensor(8.0).to(x.device))
        quant_tensor = self.features(x)
        x, scale, bit_width = self.final_pool(quant_tensor)
        x = x.view(x.size(0), -1)
        out = self.output(pack_quant_tensor(x, scale, bit_width))
        return out


def quant_mobilenet_v1(cfg):

    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False
    width_scale = float(cfg.get('MODEL', 'WIDTH_SCALE'))
    bit_width = cfg.getint('QUANT', 'BIT_WIDTH')

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(channels=channels,
                    first_stage_stride=first_stage_stride,
                    bit_width=bit_width)
    return net
