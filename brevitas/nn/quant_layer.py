# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from abc import ABCMeta
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass, field

from brevitas.quant_tensor import QuantTensor
from brevitas.proxy.parameter_quant import WeightQuantProxy

SCALING_MIN_VAL = 2.0 ** (-16)


@dataclass
class ScalingShapeConfig:
    stats_input_view_shape_impl: StatsInputViewShapeImpl
    stats_input_concat_dim: int
    stats_reduce_dim: Optional[int]
    shape: Tuple[int, ...]


@dataclass
class QuantConfig(metaclass=ABCMeta):
    quant_type: QuantType
    narrow_range: bool
    signed: bool


@dataclass
class ScalingConfig(metaclass=ABCMeta):
    restrict_value_type: Optional[RestrictValueType]
    impl_type: ScalingImplType
    min_val: Optional[float]
    stats_op: Optional[StatsOp]
    const: Optional[float]
    per_channel: bool
    stats_sigma: float


@dataclass
class BitWidthConfig(metaclass=ABCMeta):
    bit_width: Optional[int]
    impl_type: Optional[BitWidthImplType]
    restrict_value_type: Optional[RestrictValueType]
    min_val: Optional[int]
    max_val: Optional[int]
    override_pretrained: bool


@dataclass
class WeightScalingConfig(ScalingConfig):
    impl_type = ScalingImplType.STATS
    min_val = None
    stats_op = StatsOp.MAX
    const = None
    per_channel = True
    stats_sigma = 3.0


@dataclass
class WeightBitWidthConfig(BitWidthConfig):
    impl_type = BitWidthImplType.CONST
    restrict_value_type = RestrictValueType.INT
    min_val = 2
    max_val = None
    override_pretrained = False


@dataclass
class WeightQuantConfig(QuantConfig):
    scaling_config: WeightScalingConfig
    bit_width_config: WeightBitWidthConfig
    narrow_range = False
    signed: bool = field(init=False, default=True)
    ternary_threshold = 0.5


@dataclass
class BiasQuantConfig(QuantConfig):
    narrow_range = False
    signed: bool = field(init=False, default=True)


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            compute_output_scale,
            compute_output_bit_width,
            return_quant_tensor):
        self.compute_output_scale = compute_output_scale
        self.compute_output_bit_width = compute_output_bit_width
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self,
                    output,
                    output_scale,
                    output_bit_width):
        if self.return_quant_tensor:
            return QuantTensor(tensor=output, scale=output_scale, bit_width=output_bit_width)
        else:
            return output


class WeightQuantLayer(QuantLayer):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight,
            weight_quant_config,
            compute_output_scale,
            compute_output_bit_width,
            return_quant_tensor):
        super(WeightQuantLayer, self).__init__(
            compute_output_scale,
            compute_output_bit_width,
            return_quant_tensor)
        self.weight_quant = WeightQuantProxy(
            tracked_parameter_list_init=weight,
            quant_config=weight_quant_config)






