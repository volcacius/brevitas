""" Source: https://github.com/rwightman/gen-efficientnet-pytorch
Copyright 2020 Xilinx (Alessandro Pappalardo)
Copyright 2020 Ross Wightman

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

from geffnet.efficientnet_builder import decode_arch_def, round_channels, BN_EPS_TF_DEFAULT
from geffnet.efficientnet_builder import make_divisible, initialize_weight_default, initialize_weight_goog
from torch import nn

from brevitas.nn.quant_conv import PaddingType
from . import layers
from .layers.common import multisample_dropout_classify, residual_add_drop_connect, MergeBnMixin


class GenericEfficientNet(MergeBnMixin, nn.Module):

    def __init__(
            self,
            block_args,
            first_layer_weight_bit_width,
            first_layer_stride,
            first_layer_padding,
            first_act_bit_width,
            last_layer_weight_bit_width,
            pw_inp_bit_width,
            dw_inp_bit_width,
            pw_weight_bit_width,
            dw_weight_bit_width,
            avg_pool_inp_bit_width,
            avg_pool_output_bit_width,
            scaling_per_channel,
            dw_scaling_per_channel,
            scaling_stats_op,
            dw_scaling_stats_op,
            merge_bn,
            base_bit_width,
            bn_eps,
            avg_pool_kernel_size,
            channel_multiplier,
            num_classes=1000,
            in_chans=3,
            stem_size=32,
            num_features=1280,
            channel_divisor=8,
            channel_min=None,
            padding_type=None,
            dropout_rate=0.,
            dropout_samples=0,
            drop_connect_rate=0.,
            weight_init='goog'):
        super(GenericEfficientNet, self).__init__()
        self.merge_bn = merge_bn
        self.bn_eps = bn_eps
        self.dropout_rate = dropout_rate
        self.dropout_samples = dropout_samples

        stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = layers.with_defaults.make_quant_conv2d(
            in_channels=in_chans,
            out_channels=stem_size,
            kernel_size=3,
            stride=first_layer_stride,
            padding=first_layer_padding,
            padding_type=padding_type,
            bias=merge_bn,
            bit_width=first_layer_weight_bit_width,
            weight_scaling_per_output_channel=scaling_per_channel,
            weight_scaling_stats_op=scaling_stats_op,
            groups=1)
        self.bn1 = nn.Identity() if merge_bn else nn.BatchNorm2d(stem_size, eps=bn_eps)
        self.act1 = layers.with_defaults.make_quant_relu(bit_width=first_act_bit_width)
        in_chans = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier=channel_multiplier,
            channel_divisor=channel_divisor,
            channel_min=channel_min,
            padding_type=padding_type,
            bn_eps=bn_eps,
            drop_connect_rate=drop_connect_rate,
            base_bit_width=base_bit_width,
            pw_inp_bit_width=pw_inp_bit_width,
            dw_inp_bit_width=dw_inp_bit_width,
            pw_weight_bit_width=pw_weight_bit_width,
            dw_weight_bit_width=dw_weight_bit_width,
            scaling_per_channel=scaling_per_channel,
            dw_scaling_per_channel=dw_scaling_per_channel,
            scaling_stats_op=scaling_stats_op,
            dw_scaling_stats_op=dw_scaling_stats_op,
            merge_bn=merge_bn)
        self.blocks = nn.Sequential(*builder(in_chans, block_args))
        in_chans = builder.in_chs

        self.conv_head = layers.with_defaults.make_quant_conv2d(
            in_channels=in_chans,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding_type=padding_type,
            bias=merge_bn,
            bit_width=pw_weight_bit_width,
            weight_scaling_per_output_channel=scaling_per_channel,
            weight_scaling_stats_op=scaling_stats_op,
            groups=1)
        self.bn2 = nn.Identity() if merge_bn else nn.BatchNorm2d(num_features, eps=bn_eps)
        self.act2 = layers.with_defaults.make_quant_relu(
            bit_width=avg_pool_inp_bit_width,
            return_quant_tensor=True)
        self.global_pool = layers.with_defaults.make_quant_avg_pool(
            bit_width=avg_pool_output_bit_width,
            kernel_size=avg_pool_kernel_size,
            signed=False,
            stride=1)
        self.classifier = layers.with_defaults.make_quant_linear(
            in_channels=num_features,
            out_channels=num_classes,
            bias=True,
            bit_width=last_layer_weight_bit_width)

        for n, m in self.named_modules():
            if weight_init == 'goog':
                initialize_weight_goog(m, n)
            else:
                initialize_weight_default(m, n)

    def features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        out = multisample_dropout_classify(
            x,
            training=self.training,
            classifier=self.classifier,
            samples=self.dropout_samples,
            rate=self.dropout_rate)
        return out

    def conv_bn_tuples(self):
        return [(self.conv_stem, 'conv_stem', 'bn1'),
                (self.conv_head, 'conv_head', 'bn2')]


class EfficientNetBuilder:

    def __init__(
            self,
            base_bit_width,
            pw_inp_bit_width,
            dw_inp_bit_width,
            pw_weight_bit_width,
            dw_weight_bit_width,
            padding_type,
            scaling_per_channel,
            dw_scaling_per_channel,
            scaling_stats_op,
            dw_scaling_stats_op,
            merge_bn,
            bn_eps,
            channel_multiplier,
            channel_divisor,
            channel_min,
            drop_connect_rate):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.padding_type = padding_type
        self.bn_eps = bn_eps
        self.drop_connect_rate = drop_connect_rate
        self.base_bit_width = base_bit_width
        self.pw_inp_bit_width = pw_inp_bit_width
        self.dw_inp_bit_width = dw_inp_bit_width
        self.pw_weight_bit_width = pw_weight_bit_width
        self.dw_weight_bit_width = dw_weight_bit_width
        self.merge_bn = merge_bn
        self.scaling_per_channel = scaling_per_channel
        self.dw_scaling_per_channel = dw_scaling_per_channel
        self.scaling_stats_op = scaling_stats_op
        self.dw_scaling_stats_op = dw_scaling_stats_op
        self.shared_hard_tanh = None

        # updated during build
        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0

    def _round_channels(self, chs):
        return round_channels(
            chs,
            self.channel_multiplier,
            self.channel_divisor,
            self.channel_min)

    def make_shared_hard_tanh(self):
        return layers.with_defaults.make_quant_hard_tanh(
            bit_width=self.pw_inp_bit_width,
            return_quant_tensor=True)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        in_chs = self.in_chs
        out_chs = self._round_channels(ba['out_chs'])
        ba['has_residual'] = (in_chs == out_chs and ba['stride'] == 1) and not ba['noskip']
        if not ba['has_residual']:
            self.shared_hard_tanh = self.make_shared_hard_tanh()
        # This is a hack to work around mismatch in origin impl input filters for EdgeTPU
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            fake_in_chs = self._round_channels(ba['fake_in_chs'])
        else:
            fake_in_chs = 0
        if bt == 'ds':
            drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
            block = DepthwiseSeparableConv(
                in_chs=in_chs,
                out_chs=out_chs,
                pw_inp_bit_width=self.pw_inp_bit_width,
                pw_weight_bit_width=self.pw_weight_bit_width,
                dw_weight_bit_width=self.dw_weight_bit_width,
                bn_eps=self.bn_eps,
                padding_type=self.padding_type,
                drop_connect_rate=drop_connect_rate,
                shared_hard_tanh=self.shared_hard_tanh,
                pw_scaling_per_channel=self.scaling_per_channel,
                dw_scaling_per_channel=self.dw_scaling_per_channel,
                pw_scaling_stats_op=self.scaling_stats_op,
                dw_scaling_stats_op=self.dw_scaling_stats_op,
                merge_bn=self.merge_bn,
                dw_kernel_size=ba['dw_kernel_size'],
                pw_kernel_size=ba['pw_kernel_size'],
                stride=ba['stride'],
                has_residual=ba['has_residual'])
        elif bt == 'ir':
            drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
            block = InvertedResidual(
                in_chs=in_chs,
                out_chs=out_chs,
                pw_inp_bit_width= self.pw_inp_bit_width,
                dw_inp_bit_width = self.dw_inp_bit_width,
                pw_weight_bit_width = self.pw_weight_bit_width,
                dw_weight_bit_width = self.dw_weight_bit_width,
                bn_eps=self.bn_eps,
                padding_type=self.padding_type,
                drop_connect_rate=drop_connect_rate,
                shared_hard_tanh=self.shared_hard_tanh,
                pw_scaling_per_channel=self.scaling_per_channel,
                dw_scaling_per_channel=self.dw_scaling_per_channel,
                pw_scaling_stats_op=self.scaling_stats_op,
                dw_scaling_stats_op=self.dw_scaling_stats_op,
                merge_bn=self.merge_bn,
                exp_kernel_size=ba['exp_kernel_size'],
                dw_kernel_size=ba['dw_kernel_size'],
                pw_kernel_size=ba['pw_kernel_size'],
                exp_ratio=ba['exp_ratio'],
                stride=ba['stride'],
                has_residual=ba['has_residual'])
        elif bt == 'er':
            drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
            block = EdgeResidual(
                in_chs=in_chs,
                out_chs=out_chs,
                fake_in_chs=fake_in_chs,
                bit_width=self.base_bit_width,
                bn_eps=self.bn_eps,
                padding_type=self.padding_type,
                drop_connect_rate=drop_connect_rate,
                shared_hard_tanh=self.shared_hard_tanh,
                scaling_per_channel=self.scaling_per_channel,
                scaling_stats_op=self.scaling_stats_op,
                merge_bn=self.merge_bn,
                exp_kernel_size=ba['exp_kernel_size'],
                pw_kernel_size=ba['pw_kernel_size'],
                exp_ratio=ba['exp_ratio'],
                stride=ba['stride'],
                has_residual=ba['has_residual'])
        else:
            raise Exception('Uknkown block type {} while building model.'.format(bt))
        self.in_chs = out_chs  # update in_chs for arg of next block
        return block

    def _make_stack(self, stack_args):
        blocks = []
        # each stack (stage) contains a list of block arguments
        for i, ba in enumerate(stack_args):
            if i >= 1:
                # only the first block in any stack can have a stride > 1
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
            self.block_idx += 1  # incr global idx (across all stacks)
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []
        # outer list of block_args defines the stacks ('stages' by some conventions)
        for stack_idx, stack in enumerate(block_args):
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return blocks


class EdgeResidual(MergeBnMixin, nn.Module):
    """ EdgeTPU Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(
            self,
            in_chs,
            out_chs,
            bit_width,
            padding_type,
            bn_eps,
            exp_kernel_size,
            exp_ratio,
            fake_in_chs,
            stride,
            has_residual,
            pw_kernel_size,
            drop_connect_rate,
            scaling_per_channel,
            scaling_stats_op,
            shared_hard_tanh,
            merge_bn):
        super(EdgeResidual, self).__init__()
        mid_chs = make_divisible(fake_in_chs * exp_ratio) if fake_in_chs > 0 else make_divisible(in_chs * exp_ratio)
        self.has_residual = has_residual
        self.merge_bn = merge_bn
        self.bn_eps = bn_eps
        self.drop_connect_rate = drop_connect_rate
        self.shared_hard_tanh = shared_hard_tanh

        # Expansion convolution
        self.conv_exp = layers.with_defaults.make_quant_conv2d(
            in_channels=in_chs,
            out_channels=mid_chs,
            kernel_size=exp_kernel_size,
            padding_type=padding_type,
            bias=merge_bn,
            bit_width=bit_width,
            weight_scaling_per_output_channel=scaling_per_channel,
            weight_scaling_stats_op=scaling_stats_op,
            groups=1,
            stride=1)
        self.bn1 = nn.Identity() if merge_bn else nn.BatchNorm2d(mid_chs, eps=bn_eps)
        self.act1 = layers.with_defaults.make_quant_relu(bit_width=bit_width)

        # Point-wise linear projection
        self.conv_pwl = layers.with_defaults.make_quant_conv2d(
            mid_chs,
            out_chs,
            pw_kernel_size,
            stride=stride,
            padding_type=padding_type,
            bit_width=bit_width,
            weight_scaling_per_output_channel=scaling_per_channel,
            weight_scaling_stats_op=scaling_stats_op,
            groups=1,
            bias=merge_bn)
        self.bn2 = nn.Identity() if merge_bn else nn.BatchNorm2d(out_chs, eps=bn_eps)

    def conv_bn_tuples(self):
        return [(self.conv_exp, 'conv_exp', 'bn1'),
                (self.conv_pwl, 'conv_pwl', 'bn2')]

    def forward(self, x):
        residual = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)
        x = self.shared_hard_tanh(x)
        if self.has_residual:
            x = residual_add_drop_connect(x, residual, self.training, self.drop_connect_rate)
            x = self.shared_hard_tanh(x)
        return x


class DepthwiseSeparableConv(MergeBnMixin, nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            dw_kernel_size,
            pw_kernel_size,
            stride,
            padding_type,
            merge_bn,
            bn_eps,
            dw_weight_bit_width,
            pw_weight_bit_width,
            pw_inp_bit_width,
            dw_scaling_per_channel,
            pw_scaling_per_channel,
            dw_scaling_stats_op,
            pw_scaling_stats_op,
            has_residual,
            shared_hard_tanh,
            drop_connect_rate):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_residual = has_residual
        self.drop_connect_rate = drop_connect_rate
        self.merge_bn = merge_bn
        self.shared_hard_tanh = shared_hard_tanh

        self.conv_dw = layers.with_defaults.make_quant_conv2d(
            in_chs,
            in_chs,
            dw_kernel_size,
            stride=stride,
            padding_type=padding_type,
            groups=in_chs,
            bias=merge_bn,
            bit_width=dw_weight_bit_width,
            weight_scaling_per_output_channel=dw_scaling_per_channel,
            weight_scaling_stats_op=dw_scaling_stats_op)
        self.bn1 = nn.Identity() if merge_bn else nn.BatchNorm2d(in_chs, eps=bn_eps)
        self.act1 = layers.with_defaults.make_quant_relu(
            bit_width=pw_inp_bit_width)  # is input to pw conv
        self.conv_pw = layers.with_defaults.make_quant_conv2d(
            in_chs,
            out_chs,
            pw_kernel_size,
            padding_type=padding_type,
            bias=merge_bn,
            bit_width=pw_weight_bit_width,
            weight_scaling_per_output_channel=pw_scaling_per_channel,
            weight_scaling_stats_op=pw_scaling_stats_op,
            groups=1,
            stride=1)
        self.bn2 = nn.Identity() if merge_bn else nn.BatchNorm2d(out_chs, eps=bn_eps)

    def conv_bn_tuples(self):
        return [(self.conv_pw, 'conv_dw', 'bn1'),
                (self.conv_dw, 'conv_pw', 'bn2')]

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.shared_hard_tanh(x)

        if self.has_residual:
            x = residual_add_drop_connect(x, residual, self.training, self.drop_connect_rate)
            x = self.shared_hard_tanh(x)
        return x


class InvertedResidual(MergeBnMixin, nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            padding_type,
            bn_eps,
            dw_kernel_size,
            pw_inp_bit_width,
            dw_inp_bit_width,
            pw_weight_bit_width,
            dw_weight_bit_width,
            stride,
            has_residual,
            exp_ratio,
            exp_kernel_size,
            pw_kernel_size,
            pw_scaling_per_channel,
            dw_scaling_per_channel,
            pw_scaling_stats_op,
            dw_scaling_stats_op,
            drop_connect_rate,
            shared_hard_tanh,
            merge_bn):
        super(InvertedResidual, self).__init__()
        mid_chs: int = make_divisible(in_chs * exp_ratio)
        self.has_residual = has_residual
        self.drop_connect_rate = drop_connect_rate
        self.shared_hard_tanh = shared_hard_tanh
        self.merge_bn = merge_bn
        self.bn_eps = bn_eps

        # Point-wise expansion
        self.conv_pw = layers.with_defaults.make_quant_conv2d(
            in_chs,
            mid_chs,
            exp_kernel_size,
            padding_type=padding_type,
            bias=merge_bn,
            bit_width=pw_weight_bit_width,
            weight_scaling_per_output_channel=pw_scaling_per_channel,
            weight_scaling_stats_op=pw_scaling_stats_op,
            stride=1,
            groups=1)
        self.bn1 = nn.Identity() if merge_bn else nn.BatchNorm2d(mid_chs, eps=bn_eps)
        self.act1 = layers.with_defaults.make_quant_relu(
            bit_width=dw_inp_bit_width,  # is input to dw conv
            scaling_per_channel=dw_scaling_per_channel,
            per_channel_broadcastable_shape=(1, mid_chs, 1, 1))

        # Depth-wise convolution
        self.conv_dw = layers.with_defaults.make_quant_conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            padding_type=padding_type,
            groups=mid_chs,
            bias=merge_bn,
            bit_width=dw_weight_bit_width,
            weight_scaling_per_output_channel=dw_scaling_per_channel,
            weight_scaling_stats_op=dw_scaling_stats_op)
        self.bn2 = nn.Identity() if merge_bn else nn.BatchNorm2d(mid_chs, eps=bn_eps)
        self.act2 = layers.with_defaults.make_quant_relu(
            bit_width=pw_inp_bit_width)

        # Point-wise linear projection
        self.conv_pwl = layers.with_defaults.make_quant_conv2d(
            mid_chs,
            out_chs,
            pw_kernel_size,
            padding_type=padding_type,
            bias=merge_bn,
            bit_width=pw_weight_bit_width,
            weight_scaling_per_output_channel=pw_scaling_per_channel,
            weight_scaling_stats_op=pw_scaling_stats_op,
            groups=1,
            stride=1)
        self.bn3 = nn.Identity() if merge_bn else nn.BatchNorm2d(out_chs, eps=bn_eps)

    def conv_bn_tuples(self):
        return [(self.conv_pw, 'conv_pw', 'bn1'),
                (self.conv_dw, 'conv_dw', 'bn2'),
                (self.conv_pwl, 'conv_pwl', 'bn3')]

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)
        x = self.shared_hard_tanh(x)

        if self.has_residual:
            x = residual_add_drop_connect(x, residual, self.training, self.drop_connect_rate)
            x = self.shared_hard_tanh(x)

        return x


def generic_efficientnet_edge(
        hparams,
        bn_eps,
        padding_type,
        channel_multiplier,
        depth_multiplier):
    arch_def = [
        # NOTE `fc` is present to override a mismatch between stem channels and in chs not
        # present in other models
        ['er_r1_k3_s1_e4_c24_fc24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    block_args = decode_arch_def(arch_def, depth_multiplier)
    num_features = round_channels(1280, channel_multiplier, 8, None)
    stem_size = 32
    channel_multiplier = channel_multiplier
    model = GenericEfficientNet(
        block_args,
        bn_eps=bn_eps,
        padding_type=padding_type,
        num_features=num_features,
        channel_multiplier=channel_multiplier,
        stem_size=stem_size,
        first_layer_weight_bit_width=hparams.model.FIRST_LAYER_WEIGHT_BIT_WIDTH,
        first_layer_padding=hparams.model.FIRST_LAYER_PADDING,
        first_layer_stride=hparams.model.FIRST_LAYER_STRIDE,
        last_layer_weight_bit_width=hparams.model.LAST_LAYER_WEIGHT_BIT_WIDTH,
        dw_scaling_per_channel=hparams.model.DW_SCALING_PER_CHANNEL,
        scaling_per_channel=hparams.model.SCALING_PER_CHANNEL,
        dw_scaling_stats_op=hparams.model.DW_SCALING_STATS_OP,
        scaling_stats_op=hparams.model.SCALING_STATS_OP,
        base_bit_width=hparams.model.BASE_BIT_WIDTH,
        dropout_rate=hparams.dropout.RATE,
        dropout_samples=hparams.dropout.SAMPLES,
        drop_connect_rate=hparams.drop_connect.RATE,
        avg_pool_kernel_size=hparams.model.AVG_POOL_KERNEL_SIZE,
        merge_bn=hparams.model.MERGE_BN,
        dw_inp_bit_width=hparams.model.DW_INP_BIT_WIDTH,
        pw_inp_bit_width=hparams.model.PW_INP_BIT_WIDTH,
        dw_weight_bit_width=hparams.model.DW_WEIGHT_BIT_WIDTH,
        pw_weight_bit_width=hparams.model.PW_WEIGHT_BIT_WIDTH,
        avg_pool_inp_bit_width=hparams.model.AVG_POOL_INP_BIT_WIDTH,
        avg_pool_output_bit_width=hparams.model.AVG_POOL_OUTPUT_BIT_WIDTH,
        first_act_bit_width=hparams.model.PW_INP_BIT_WIDTH)  # second conv is pw
    return model


def generic_efficientnet_lite(
        hparams,
        bn_eps,
        padding_type,
        channel_multiplier,
        depth_multiplier):
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    block_args = decode_arch_def(arch_def, depth_multiplier, fix_first_last=True)
    num_features = 1280
    stem_size = 32
    channel_multiplier = channel_multiplier
    model = GenericEfficientNet(
        block_args,
        bn_eps=bn_eps,
        padding_type=padding_type,
        num_features=num_features,
        channel_multiplier=channel_multiplier,
        stem_size=stem_size,
        first_layer_weight_bit_width=hparams.model.FIRST_LAYER_WEIGHT_BIT_WIDTH,
        first_layer_padding=hparams.model.FIRST_LAYER_PADDING,
        first_layer_stride=hparams.model.FIRST_LAYER_STRIDE,
        last_layer_weight_bit_width=hparams.model.LAST_LAYER_WEIGHT_BIT_WIDTH,
        dw_scaling_per_channel=hparams.model.DW_SCALING_PER_CHANNEL,
        scaling_per_channel=hparams.model.SCALING_PER_CHANNEL,
        dw_scaling_stats_op=hparams.model.DW_SCALING_STATS_OP,
        scaling_stats_op=hparams.model.SCALING_STATS_OP,
        base_bit_width=hparams.model.BASE_BIT_WIDTH,
        dropout_rate=hparams.dropout.RATE,
        dropout_samples=hparams.dropout.SAMPLES,
        drop_connect_rate=hparams.drop_connect.RATE,
        avg_pool_kernel_size=hparams.model.AVG_POOL_KERNEL_SIZE,
        merge_bn=hparams.model.MERGE_BN,
        dw_inp_bit_width=hparams.model.DW_INP_BIT_WIDTH,
        pw_inp_bit_width=hparams.model.PW_INP_BIT_WIDTH,
        dw_weight_bit_width=hparams.model.DW_WEIGHT_BIT_WIDTH,
        pw_weight_bit_width=hparams.model.PW_WEIGHT_BIT_WIDTH,
        avg_pool_inp_bit_width=hparams.model.AVG_POOL_INP_BIT_WIDTH,
        avg_pool_output_bit_width=hparams.model.AVG_POOL_OUTPUT_BIT_WIDTH,
        first_act_bit_width=hparams.model.DW_INP_BIT_WIDTH)  # second conv is dw
    return model


def quant_tf_efficientnet_es(hparams):
    """ EfficientNet-Edge Small. Tensorflow compatible variant  """
    model = generic_efficientnet_edge(
        hparams,
        bn_eps=BN_EPS_TF_DEFAULT,
        padding_type=PaddingType.SAME,
        channel_multiplier=1.0,
        depth_multiplier=1.0)
    return model


def quant_tf_efficientnet_em(hparams):
    """ EfficientNet-Edge-Medium. Tensorflow compatible variant  """
    model = generic_efficientnet_edge(
        hparams,
        bn_eps=BN_EPS_TF_DEFAULT,
        padding_type=PaddingType.SAME,
        channel_multiplier=1.0,
        depth_multiplier=1.1)
    return model


def quant_tf_efficientnet_el(hparams):
    """ EfficientNet-Edge-Large. Tensorflow compatible variant  """
    model = generic_efficientnet_edge(
        hparams,
        bn_eps=BN_EPS_TF_DEFAULT,
        padding_type=PaddingType.SAME,
        channel_multiplier=1.2,
        depth_multiplier=1.4)
    return model


def quant_tf_efficientnet_lite0(hparams):
    model = generic_efficientnet_lite(
        hparams,
        bn_eps=BN_EPS_TF_DEFAULT,
        padding_type=PaddingType.SAME,
        channel_multiplier=1.0,
        depth_multiplier=1.0)
    return model


def quant_tf_efficientnet_lite1(hparams):
    model = generic_efficientnet_lite(
        hparams,
        channel_multiplier=1.0,
        depth_multiplier=1.1,
        bn_eps = BN_EPS_TF_DEFAULT,
        padding_type = PaddingType.SAME)
    return model


def quant_tf_efficientnet_lite2(hparams):
    model = generic_efficientnet_lite(
        hparams,
        channel_multiplier=1.1,
        depth_multiplier=1.2,
        bn_eps=BN_EPS_TF_DEFAULT,
        padding_type=PaddingType.SAME)
    return model


def quant_tf_efficientnet_lite3(hparams):
    model = generic_efficientnet_lite(
        hparams,
        channel_multiplier=1.2,
        depth_multiplier=1.4,
        bn_eps=BN_EPS_TF_DEFAULT,
        padding_type=PaddingType.SAME)
    return model


def quant_tf_efficientnet_lite4(hparams):
    model = generic_efficientnet_lite(
        hparams,
        channel_multiplier=1.4,
        depth_multiplier=1.8,
        bn_eps=BN_EPS_TF_DEFAULT,
        padding_type=PaddingType.SAME)
    return model