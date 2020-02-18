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

from torch import nn
from torch.nn import Sequential

from . import layers
from .layers.common import multisample_dropout_classify, residual_add_drop_connect, MergeBnMixin


class DwsConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 merge_bn,
                 stride,
                 weight_bit_width,
                 activation_bit_width,
                 weight_scaling_per_output_channel,
                 pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            merge_bn=merge_bn,
            kernel_size=3,
            padding=1,
            stride=stride,
            weight_bit_width=weight_bit_width,
            act_bit_width=activation_bit_width,
            weight_scaling_per_output_channel=weight_scaling_per_output_channel)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            merge_bn=merge_bn,
            kernel_size=1,
            padding=0,
            weight_bit_width=weight_bit_width,
            activation_bit_width=activation_bit_width,
            weight_scaling_per_output_channel=weight_scaling_per_output_channel,
            activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvBlock(MergeBnMixin, nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 weight_bit_width,
                 activation_bit_width,
                 padding,
                 merge_bn,
                 stride=1,
                 groups=1,
                 bn_eps=1e-5,
                 weight_scaling_per_output_channel=True,
                 activation_scaling_per_channel=False):
        super(ConvBlock, self).__init__()
        self.merge_bn = merge_bn
        self.bn_eps = bn_eps
        self.conv = layers.with_defaults.make_quant_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_scaling_per_output_channel=weight_scaling_per_output_channel,
            bit_width=weight_bit_width)
        self.bn = nn.Identity() if merge_bn else nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = layers.with_defaults.make_quant_relu(
            bit_width=activation_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=activation_scaling_per_channel,
            return_quant_tensor=True)

    def conv_bn_tuples(self):
        return [(self.conv, 'conv', 'bn')]

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class MobileNet(nn.Module):

    def __init__(self,
                 channels,
                 first_layer_stride,
                 first_layer_weight_bit_width,
                 first_layer_padding,
                 first_stage_stride,
                 weight_scaling_per_channel,
                 activation_scaling_per_channel,
                 weight_bit_width,
                 activation_bit_width,
                 dropout_rate,
                 dropout_samples,
                 merge_bn,
                 in_channels=3,
                 num_classes=1000):
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]
        self.dropout_rate = dropout_rate
        self.dropout_samples = dropout_samples
        self.features = Sequential()
        init_block = ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=first_layer_stride,
            padding=first_layer_padding,
            weight_bit_width=first_layer_weight_bit_width,
            weight_scaling_per_output_channel=weight_scaling_per_channel,
            activation_scaling_per_channel=activation_scaling_per_channel,
            activation_bit_width=weight_bit_width,
            merge_bn=merge_bn)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1 and activation_scaling_per_channel
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                mod = DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    weight_bit_width=weight_bit_width,
                    activation_bit_width=activation_bit_width,
                    weight_scaling_per_output_channel=weight_scaling_per_channel,
                    pw_activation_scaling_per_channel=pw_activation_scaling_per_channel,
                    merge_bn=merge_bn)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = layers.with_defaults.make_quant_avg_pool(
            kernel_size=7,
            stride=1,
            signed=False,
            bit_width=activation_bit_width)
        self.output = layers.with_defaults.make_quant_linear(
            in_channels,
            num_classes,
            bias=True,
            enable_bias_quant=True,
            bit_width=weight_bit_width)

    def forward(self, x):
        quant_tensor = self.features(x)
        x = self.final_pool(quant_tensor)
        out = multisample_dropout_classify(
            x,
            training=self.training,
            classifier=self.output,
            samples=self.dropout_samples,
            rate=self.dropout_rate)
        return out


def quant_mobilenet_v1(hparams):

    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False

    if hparams.model.WIDTH_SCALE != 1.0:
        channels = [[int(cij * hparams.model.WIDTH_SCALE) for cij in ci] for ci in channels]

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        first_layer_weight_bit_width=hparams.model.FIRST_LAYER_WEIGHT_BIT_WIDTH,
        first_layer_padding=hparams.model.FIRST_LAYER_PADDING,
        first_layer_stride=hparams.model.FIRST_LAYER_STRIDE,
        weight_scaling_per_channel=hparams.model.WEIGHT_SCALING_PER_CHANNEL,
        activation_scaling_per_channel=hparams.model.ACTIVATION_SCALING_PER_CHANNEL,
        weight_bit_width=hparams.model.WEIGHT_BIT_WIDTH,
        activation_bit_width=hparams.model.ACTIVATION_BIT_WIDTH,
        dropout_rate=hparams.dropout.RATE,
        dropout_samples=hparams.dropout.SAMPLES,
        merge_bn=hparams.model.MERGE_BN)
    return net
