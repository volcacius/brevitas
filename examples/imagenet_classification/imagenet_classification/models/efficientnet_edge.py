""" Source: https://github.com/rwightman/gen-efficientnet-pytorch
Copyright 2019 Xilinx (Alessandro Pappalardo)
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

from brevitas.nn.quant_conv import PaddingType
from geffnet.efficientnet_builder import decode_arch_def, round_channels, drop_connect, BN_EPS_TF_DEFAULT
from geffnet.efficientnet_builder import make_divisible, initialize_weight_default, initialize_weight_goog
from torch import nn

from . import layers
from .layers.common import multisample_dropout_classify


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


def generic_efficientnet_edge(hparams,
                              bn_eps,
                              padding_type,
                              channel_multiplier=1.0,
                              depth_multiplier=1.0):
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
        bit_width=hparams.model.BIT_WIDTH,
        dropout_rate=hparams.dropout.RATE,
        dropout_samples=hparams.dropout.SAMPLES)
    return model


class GenericEfficientNet(nn.Module):

    def __init__(self,
                 block_args,
                 first_layer_weight_bit_width,
                 first_layer_stride,
                 first_layer_padding,
                 bit_width,
                 bn_eps,
                 num_classes=1000,
                 in_chans=3,
                 stem_size=32,
                 num_features=1280,
                 channel_multiplier=1.0,
                 channel_divisor=8,
                 channel_min=None,
                 padding_type=None,
                 dropout_rate=0.,
                 dropout_samples=0,
                 drop_connect_rate=0.,
                 weight_init='goog'):
        super(GenericEfficientNet, self).__init__()
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
            bias=False,
            bit_width=first_layer_weight_bit_width,
            groups=1)
        self.bn1 = nn.BatchNorm2d(stem_size, eps=bn_eps)
        self.act1 = layers.with_defaults.make_quant_relu(bit_width=bit_width)
        in_chans = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier=channel_multiplier,
            channel_divisor=channel_divisor,
            channel_min=channel_min,
            padding_type=padding_type,
            bn_eps=bn_eps,
            drop_connect_rate=drop_connect_rate,
            bit_width=bit_width)
        self.blocks = nn.Sequential(*builder(in_chans, block_args))
        in_chans = builder.in_chs

        self.conv_head = layers.with_defaults.make_quant_conv2d(
            in_channels=in_chans,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding_type=padding_type,
            bias=False,
            bit_width=bit_width,
            groups=1)
        self.bn2 = nn.BatchNorm2d(num_features, eps=bn_eps)
        self.act2 = layers.with_defaults.make_quant_relu(bit_width=bit_width, return_quant_tensor=True)
        self.global_pool = layers.with_defaults.make_quant_avg_pool(
            bit_width=bit_width,
            kernel_size=7,
            signed=False,
            stride=1)
        self.classifier = layers.with_defaults.make_quant_linear(
            in_channels=num_features,
            out_channels=num_classes,
            bias=True,
            enable_bias_quant=True,
            bit_width=bit_width,
            weight_scaling_per_output_channel=False)

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


class EfficientNetBuilder:

    def __init__(self,
                 bit_width,
                 padding_type,
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
        self.bit_width = bit_width
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
            bit_width=self.bit_width,
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
        if bt == 'ir':
            drop_connect_rate = self.drop_connect_rate * self.block_idx / self.block_count
            block = InvertedResidual(
                in_chs=in_chs,
                out_chs=out_chs,
                bit_width=self.bit_width,
                bn_eps=self.bn_eps,
                padding_type=self.padding_type,
                drop_connect_rate=drop_connect_rate,
                shared_hard_tanh=self.shared_hard_tanh,
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
                bit_width=self.bit_width,
                bn_eps=self.bn_eps,
                padding_type=self.padding_type,
                drop_connect_rate=drop_connect_rate,
                shared_hard_tanh=self.shared_hard_tanh,
                exp_kernel_size=ba['exp_kernel_size'],
                pw_kernel_size=ba['pw_kernel_size'],
                exp_ratio=ba['exp_ratio'],
                stride=ba['stride'],
                has_residual=ba['has_residual'])
        else:
            raise Exception('Uknkown block type {} while building model.'.format(bt))
        self.in_chs = ba['out_chs']  # update in_chs for arg of next block
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


class EdgeResidual(nn.Module):
    """ EdgeTPU Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self,
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
                 shared_hard_tanh):
        super(EdgeResidual, self).__init__()
        mid_chs = make_divisible(fake_in_chs * exp_ratio) if fake_in_chs > 0 else make_divisible(in_chs * exp_ratio)
        self.has_residual = has_residual
        self.drop_connect_rate = drop_connect_rate
        self.shared_hard_tanh = shared_hard_tanh

        # Expansion convolution
        self.conv_exp = layers.with_defaults.make_quant_conv2d(
            in_channels=in_chs,
            out_channels=mid_chs,
            kernel_size=exp_kernel_size,
            padding_type=padding_type,
            bias=False,
            bit_width=bit_width,
            groups=1,
            stride=1)
        self.bn1 = nn.BatchNorm2d(mid_chs, eps=bn_eps)
        self.act1 = layers.with_defaults.make_quant_relu(bit_width=bit_width)

        # Point-wise linear projection
        self.conv_pwl = layers.with_defaults.make_quant_conv2d(
            mid_chs,
            out_chs,
            pw_kernel_size,
            stride=stride,
            padding_type=padding_type,
            bit_width=bit_width,
            groups=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs, eps=bn_eps)

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
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
            x = self.shared_hard_tanh(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self,
                 in_chs,
                 out_chs,
                 padding_type,
                 bn_eps,
                 bit_width,
                 dw_kernel_size,
                 stride,
                 has_residual,
                 exp_ratio,
                 exp_kernel_size,
                 pw_kernel_size,
                 drop_connect_rate,
                 shared_hard_tanh):
        super(InvertedResidual, self).__init__()
        mid_chs: int = make_divisible(in_chs * exp_ratio)
        self.has_residual = has_residual
        self.drop_connect_rate = drop_connect_rate
        self.shared_hard_tanh = shared_hard_tanh

        # Point-wise expansion
        self.conv_pw = layers.with_defaults.make_quant_conv2d(
            in_chs,
            mid_chs,
            exp_kernel_size,
            padding_type=padding_type,
            bias=False,
            bit_width=bit_width,
            stride=1,
            groups=1)
        self.bn1 = nn.BatchNorm2d(mid_chs, eps=bn_eps)
        self.act1 = layers.with_defaults.make_quant_relu(
            bit_width=bit_width,
            scaling_per_channel=True,
            per_channel_broadcastable_shape=(1, mid_chs, 1, 1))

        # Depth-wise convolution
        self.conv_dw = layers.with_defaults.make_quant_conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            padding_type=padding_type,
            groups=mid_chs,
            bias=False,
            bit_width=bit_width)
        self.bn2 = nn.BatchNorm2d(mid_chs, eps=bn_eps)
        self.act2 = layers.with_defaults.make_quant_relu(bit_width=bit_width)

        # Point-wise linear projection
        self.conv_pwl = layers.with_defaults.make_quant_conv2d(
            mid_chs,
            out_chs,
            pw_kernel_size,
            padding_type=padding_type,
            bias=False,
            bit_width=bit_width,
            groups=1,
            stride=1)
        self.bn3 = nn.BatchNorm2d(out_chs, eps=bn_eps)

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
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
            x = self.shared_hard_tanh(x)

        return x
