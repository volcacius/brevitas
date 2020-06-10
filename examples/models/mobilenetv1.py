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
NUM_LAYERS = 28
STOP_PRUNING = 'layer0_block0_pw'


class DwsConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bit_width,
                 pw_weight_bit_width,
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
                                 weight_bit_width=pw_weight_bit_width,
                                 act_bit_width=bit_width,
                                 activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def export(self, name_prefix, simd_dw, simd_pw, pe_dw, pe_pw, in_ch_pruning_mask, enable_out_ch_pruning):
        export_list = []
        export_tuple, out_ch_pruning_mask = self.dw_conv.export(
            name_prefix + '_dw',
            simd_dw,
            pe_dw,
            in_ch_pruning_mask=in_ch_pruning_mask,
            enable_out_ch_pruning=enable_out_ch_pruning)
        export_list.extend(export_tuple)
        if name_prefix + '_pw' == STOP_PRUNING:
            enable_out_ch_pruning = False
        export_tuple, out_ch_pruning_mask = self.pw_conv.export(
            name_prefix + '_pw',
            simd_pw,
            pe_pw,
            in_ch_pruning_mask=out_ch_pruning_mask,
            enable_out_ch_pruning=enable_out_ch_pruning)
        export_list.extend(export_tuple)
        return export_list, out_ch_pruning_mask

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
                 mean=None,
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
        self.mean = mean
        self.int_input = None
        self.int_acc = None
        self.input_2d_shape = None
        self.output_2d_shape = None
        self.acc_scale = None
        self.acc_bit_width = None
        self.output_bit_width = None
        self.preprocessing_bias = None

    def export(self, name_prefix, simd, pe, in_ch_pruning_mask, enable_out_ch_pruning):
        bias_factor_init = 0.0
        if self.conv.bias:
            bias_factor_init += self.conv.bias
        if self.preprocessing_bias is not None:
            bias_factor_init += self.preprocessing_bias
        factors_tuple = scale_bias_fusion(
            self.bn,
            scale_factor_init=self.acc_scale,
            bias_factor_init=bias_factor_init)
        acc_scale_factor, acc_bias_factor, weight_sign_factor = factors_tuple
        if self.mean is not None:
            acc_scale_factor = acc_scale_factor / 255.0

        # Out channel pruning mask, start with the one based on weights only
        out_ch_pruning_mask = None
        if enable_out_ch_pruning:
            out_ch_pruning_mask = weight_matrix_conv_pruning_mask(self.conv)

        # Threshold
        # Update out channel pruning mask with the one from thresholds too
        threshold, out_ch_pruning_mask = hls_threshold_string(
            self.activation,
            hls_var_name='{}_threshold'.format(name_prefix.lower()),
            acc_bit_width=self.acc_bit_width,
            acc_scale_factor=acc_scale_factor,
            acc_bias_factor=acc_bias_factor,
            output_bit_width=self.output_bit_width,
            out_ch_pruning_mask=out_ch_pruning_mask,
            pe=pe,
            enable_pruning=enable_out_ch_pruning)
        # Weight
        weight_bit_width_impl = self.conv.weight_quant.tensor_quant.msb_clamp_bit_width_impl
        weight_bit_width = weight_bit_width_impl(getattr(self.conv.weight_quant, ZERO_HW_SENTINEL_NAME))
        weight_bit_width = weight_bit_width.int().item()
        conv_weight = hls_weight_string_conv(
            self.conv,
            in_ch_pruning_mask=in_ch_pruning_mask,
            out_ch_pruning_mask=out_ch_pruning_mask,
            weight_bit_width=weight_bit_width,
            hls_var_name='{}_weight'.format(name_prefix.lower()),
            sign_factor=weight_sign_factor,
            simd=simd,
            pe=pe)
        # Config
        config_list = hls_config_string_conv(
            self.conv,
            name_prefix.upper(),
            in_ch_pruning_mask=in_ch_pruning_mask,
            out_ch_pruning_mask=out_ch_pruning_mask,
            weight_bit_width=weight_bit_width,
            output_bit_width=self.output_bit_width,
            input_2d_shape=self.input_2d_shape,
            output_2d_shape=self.output_2d_shape,
            simd=simd,
            pe=pe)
        # Int inp/output
        int_input = self.int_input.detach().cpu().numpy()
        if in_ch_pruning_mask is not None:
            int_input = int_input[:, in_ch_pruning_mask]
        int_acc = self.int_acc.detach().cpu().numpy()
        if out_ch_pruning_mask is not None:
            int_acc = int_acc[:, out_ch_pruning_mask]
        int_input = (name_prefix.lower() + '_int_input', int_input)
        int_acc = (name_prefix.lower() + '_int_acc', int_acc)
        # Return as a list of a single tuple
        export_tuple = (name_prefix, conv_weight), (name_prefix, threshold), config_list, int_input, int_acc
        return [export_tuple], out_ch_pruning_mask

    def forward(self, x):
        inp = x
        acc = self.conv(x)
        x = self.bn(acc.tensor)
        x = self.activation(x)
        if EXPORT:
            self.int_input = torch.round(inp.tensor / inp.scale).int()
            # Need to incorporate the change of sign coming from batch norm
            bn_sign = (self.bn.weight.view(-1).sign() * self.bn.running_var.view(-1).sign()).view(1, -1, 1, 1)
            self.int_acc = bn_sign * torch.round(acc.tensor / acc.scale).int()
            self.input_2d_shape = (inp.tensor.shape[2], inp.tensor.shape[3])
            self.acc_scale = acc.scale.detach()
            self.acc_bit_width = acc.bit_width.int().item()
            self.output_bit_width = x.bit_width.int().item()
            self.output_2d_shape = (x.tensor.shape[2], x.tensor.shape[3])
            if self.mean is not None:
                preproc_bias_input = - inp.scale * torch.tensor(self.mean, device=x.tensor.device).view(1, -1, 1, 1)
                preproc_bias_input_shape = (inp.tensor.shape[0], inp.tensor.shape[1], *self.conv.kernel_size)
                preproc_bias_input_expanded = preproc_bias_input.expand(preproc_bias_input_shape)
                preproc_bias = self.conv(pack_quant_tensor(preproc_bias_input_expanded, 0.0, 0)).tensor
                self.preprocessing_bias = preproc_bias
                int_input = (inp.tensor - preproc_bias_input.expand_as(inp.tensor)) * 255.0 / inp.scale
                int_acc = bn_sign * (acc.tensor - preproc_bias.expand_as(acc.tensor)) * 255.0 / acc.scale
                self.int_input = torch.round(int_input).int()
                self.int_acc = torch.round(int_acc).int()
        return x


class MobileNet(nn.Module):

    def __init__(self,
                 channels,
                 first_stage_stride,
                 bit_width,
                 other_pw_weight_bit_width,
                 simd_list,
                 pe_list,
                 in_channels=3,
                 num_classes=1000,
                 mean=None,
                 std=None):
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]
        self.std = std
        self.features = Sequential()
        init_block = ConvBlock(in_channels=in_channels,
                               out_channels=init_block_channels,
                               kernel_size=3,
                               stride=2,
                               weight_bit_width=FIRST_LAYER_BIT_WIDTH,
                               activation_scaling_per_channel=True,
                               act_bit_width=bit_width,
                               mean=mean)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                if (i == 3 and (j == 1 or j == 5)) or (i == 4 and (j == 0 or j == 1)):
                    pwwbw = other_pw_weight_bit_width
                else:
                    pwwbw = bit_width
                mod = DwsConvBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   stride=stride,
                                   bit_width=bit_width,
                                   pw_weight_bit_width=pwwbw,
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
        self.avg_pool_int_input = None
        self.fc_int_input = None
        self.fc_int_output = None
        self.fc_int_bias = None
        self.avg_pool_post_sum_bit_width = None
        self.avg_pool_output_bit_width = None
        self.fc_weight_bit_width = None
        self.fc_output_bit_width = None
        self.fc_bias_bit_width = None
        self.simd_list = simd_list
        self.pe_list = pe_list

    def export(self, input_sample):
        self.eval()
        with torch.no_grad():
            self.forward(input_sample)
        export_list = []
        in_ch_pruning_mask = None
        enable_out_ch_pruning = True
        name_prefix = 'conv0'
        if name_prefix == STOP_PRUNING:
            enable_out_ch_pruning = False
        export_tuple, out_ch_pruning_mask = self.features[0].export(
            in_ch_pruning_mask=in_ch_pruning_mask,
            name_prefix=name_prefix,
            simd=self.simd_list[0],
            pe=self.pe_list[0],
            enable_out_ch_pruning=enable_out_ch_pruning)
        export_list.extend(export_tuple)
        export_index = 1
        in_ch_pruning_mask = out_ch_pruning_mask
        for i, stage in enumerate(self.features[1:]):
            for j, block in enumerate(stage):
                name_prefix = 'layer{}_block{}'.format(i, j)
                export_tuple, out_ch_pruning_mask = block.export(
                    name_prefix=name_prefix,
                    in_ch_pruning_mask=in_ch_pruning_mask,
                    enable_out_ch_pruning=enable_out_ch_pruning,
                    simd_dw=self.simd_list[export_index],
                    simd_pw=self.simd_list[export_index + 1],
                    pe_dw=self.pe_list[export_index],
                    pe_pw=self.pe_list[export_index + 1])
                export_list.extend(export_tuple)
                export_index += 2
                in_ch_pruning_mask = out_ch_pruning_mask
                if name_prefix in STOP_PRUNING:
                    enable_out_ch_pruning = False
                if not enable_out_ch_pruning:
                    in_ch_pruning_mask = None
        export_list_list = [list(i) for i in zip(*export_list)]
        weight_list, threshold_list, config_list, int_input_list, int_acc_list = export_list_list
        avg_pool_int_input = self.avg_pool_int_input.detach().cpu().numpy()
        fc_int_input = self.fc_int_input.detach().cpu().numpy()
        if in_ch_pruning_mask is not None:
            avg_pool_int_input = avg_pool_int_input[:, in_ch_pruning_mask]
            fc_int_input = fc_int_input[:,  in_ch_pruning_mask]
        int_input_list.append(('avg_pool_int_input', avg_pool_int_input))
        int_input_list.append(('fc_int_input', fc_int_input))
        int_input_list.append(('fc_int_output', self.fc_int_output.detach().cpu().numpy()))
        # FC
        name_prefix = 'fc'
        fc_weight = hls_weight_string_fc(
            self.output,
            in_ch_pruning_mask=in_ch_pruning_mask,
            simd=self.simd_list[export_index],
            pe=self.pe_list[export_index],
            weight_bit_width=self.fc_weight_bit_width,
            hls_var_name='{}_weight'.format(name_prefix.lower()))
        weight_list.append((name_prefix, fc_weight))
        fc_bias = hls_bias_string_fc(
            self.fc_int_bias,
            pe=self.pe_list[export_index],
            bias_bit_width=self.fc_bias_bit_width,
            output_bit_width=self.fc_output_bit_width,
            hls_var_name='{}_bias'.format(name_prefix.lower()))
        threshold_list.append((name_prefix, fc_bias))
        # Config
        fc_config_list = hls_config_string_fc(
            self.output,
            name_prefix.upper(),
            in_ch_pruning_mask=in_ch_pruning_mask,
            simd=self.simd_list[export_index],
            pe=self.pe_list[export_index],
            weight_bit_width=self.fc_weight_bit_width,
            output_bit_width=self.fc_output_bit_width)
        config_list.append(define('POST_SUM_WIDTH_AVG_POOL', self.avg_pool_post_sum_bit_width))
        config_list.append(define('TRUNCATE_WIDTH_AVG_POOL', self.avg_pool_post_sum_bit_width - self.avg_pool_output_bit_width))
        config_list.append(define('OUTPUT_WIDTH_AVG_POOL', self.avg_pool_output_bit_width))
        config_list.append(fc_config_list)

        return weight_list, threshold_list, config_list, int_input_list, int_acc_list

    def forward(self, x):
        x = pack_quant_tensor(x, 1.0 / self.std, torch.tensor(8.0).to(x.device))
        quant_tensor = self.features(x)
        x, scale, bit_width = self.final_pool(quant_tensor)
        x = x.view(x.size(0), -1)
        out = self.output(pack_quant_tensor(x, scale, bit_width))
        if EXPORT:
            self.avg_pool_int_input = torch.round(quant_tensor.tensor / quant_tensor.scale).int()
            self.fc_int_input = torch.round(x / scale).int()
            self.fc_int_output = torch.round(out.tensor / out.scale).int()
            self.avg_pool_post_sum_bit_width = self.final_pool.max_output_bit_width(quant_tensor.bit_width).int().item()
            self.avg_pool_output_bit_width = bit_width.int().item()
            self.fc_output_bit_width = out.bit_width.int().item()
            fc_weight_bit_width_impl = self.output.weight_quant.tensor_quant.msb_clamp_bit_width_impl
            fc_weight_bit_width = fc_weight_bit_width_impl(getattr(self.output.weight_quant, ZERO_HW_SENTINEL_NAME))
            self.fc_weight_bit_width = fc_weight_bit_width.int().item()
            fc_bias = self.output.bias_quant(self.output.bias, out.scale, self.output.max_output_bit_width(bit_width, fc_weight_bit_width))
            self.fc_int_bias = torch.round(fc_bias[0] / fc_bias[1]).int()
            self.fc_bias_bit_width = fc_bias[2].int().item()
        return out.tensor


def quant_mobilenet_v1(cfg):

    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False
    width_scale = float(cfg.get('MODEL', 'WIDTH_SCALE'))
    bit_width = cfg.getint('QUANT', 'BIT_WIDTH')
    mean = [float(cfg.get('PREPROCESS', 'MEAN_0')), float(cfg.get('PREPROCESS', 'MEAN_1')),
            float(cfg.get('PREPROCESS', 'MEAN_2'))]
    std = float(cfg.get('PREPROCESS', 'STD_0'))
    other_pw_weight_bit_width = cfg.getint('QUANT', 'OTHER_PW_WEIGHT_BIT_WIDTH')
    try:
        simd_list = list(map(int, cfg.get('EXPORT', 'SIMD').split(',')))
        pe_list = list(map(int, cfg.get('EXPORT', 'PE').split(',')))
    except:
        simd_list = pe_list = [None for i in range(NUM_LAYERS)]
    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(channels=channels,
                    first_stage_stride=first_stage_stride,
                    bit_width=bit_width,
                    mean=mean,
                    std=std,
                    simd_list=simd_list,
                    pe_list=pe_list,
                    other_pw_weight_bit_width=other_pw_weight_bit_width)
    return net
