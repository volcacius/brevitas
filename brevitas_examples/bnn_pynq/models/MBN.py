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

from torch.nn import Module, BatchNorm2d, AvgPool2d, Sequential
from .tensor_norm import TensorNorm
from .common import get_quant_type, get_stats_op
from brevitas.quant_tensor import *

from brevitas_examples.imagenet_classification.models.common import *

SCALING_MIN_VAL = 1e-8


class DwsConvBlock(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 weight_bit_width,
                 act_bit_width,
                 pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(in_channels=in_channels,
                                 out_channels=in_channels,
                                 groups=in_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=stride,
                                 weight_bit_width=weight_bit_width,
                                 act_bit_width=act_bit_width)
        self.pw_conv = ConvBlock(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 padding=0,
                                 weight_bit_width=weight_bit_width,
                                 act_bit_width=act_bit_width,
                                 activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvBlock(Module):

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
                                      weight_scaling_min_val=SCALING_MIN_VAL,
                                      weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
                                      weight_scaling_stats_op=get_stats_op(weight_bit_width),
                                      weight_quant_type=get_quant_type(weight_bit_width),
                                      bit_width=weight_bit_width)
        self.bn = BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = make_quant_hard_tanh(bit_width=act_bit_width,
                                               per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                               scaling_per_channel=activation_scaling_per_channel,
                                               quant_type=get_quant_type(act_bit_width),
                                               scaling_impl_type=ScalingImplType.STATS,
                                               scaling_min_val=SCALING_MIN_VAL,
                                               threshold=1.0,
                                               return_quant_tensor=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class MBN(Module):

    def __init__(self,
                 channels,
                 first_stage_stride=False,
                 num_classes=10,
                 weight_bit_width=None,
                 act_bit_width=None,
                 in_ch=3):
        super(MBN, self).__init__()
        init_block_channels = channels[0][0]

        self.features = Sequential()
        init_block = ConvBlock(in_channels=in_ch,
                               out_channels=init_block_channels,
                               kernel_size=3,
                               stride=1,
                               weight_bit_width=weight_bit_width,
                               activation_scaling_per_channel=True,
                               act_bit_width=act_bit_width)
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
                                   weight_bit_width=weight_bit_width,
                                   act_bit_width=act_bit_width,
                                   pw_activation_scaling_per_channel=pw_activation_scaling_per_channel)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = AvgPool2d(kernel_size=2, stride=1)
        self.final_act = make_quant_hard_tanh(bit_width=act_bit_width,
                                              quant_type=get_quant_type(act_bit_width),
                                              threshold=1.0,
                                              scaling_impl_type=ScalingImplType.STATS,
                                              scaling_min_val=SCALING_MIN_VAL,
                                              return_quant_tensor=False)
        self.output = make_quant_linear(in_channels, num_classes,
                                        bias=False,
                                        bit_width=weight_bit_width,
                                        weight_quant_type=get_quant_type(weight_bit_width),
                                        weight_scaling_min_val=SCALING_MIN_VAL,
                                        weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
                                        weight_scaling_per_output_channel=False)

    def clip_weights(self, min_val, max_val):
        pass

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x, _, _ = self.features(x)
        x = self.final_pool(x)
        x = self.final_act(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        return out


def mbn(cfg):
    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    width_scale = cfg.getfloat('MODEL', 'WIDTH_SCALE')
    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
    net = MBN(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              num_classes=num_classes,
              in_ch=in_channels,
              channels=channels)
    return net
