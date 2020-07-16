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
from typing import Type, Union, Callable

from torch.nn import Module
from dependencies import Injector

from brevitas.proxy.runtime_quant import IdentityQuantProxy, ActQuantProxy
from .base import QuantActMixin, QuantLayerMixin


class QuantInputMixin(QuantActMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_quant: Union[IdentityQuantProxy, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        QuantActMixin.__init__(
            self,
            act_impl=None,
            act_quant=act_quant,
            update_injector=update_injector,
            proxy_impl=IdentityQuantProxy,
            proxy_prefix='input_',
            kwargs_prefix='input_',
            **kwargs)

    def quant_input_scale(self):
        return self.act_quant.scale()


class QuantOutputMixin(QuantActMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_quant: Union[IdentityQuantProxy, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        QuantActMixin.__init__(
            self,
            act_impl=None,
            act_quant=act_quant,
            update_injector=update_injector,
            proxy_impl=IdentityQuantProxy,
            proxy_prefix='output_',
            kwargs_prefix='output_',
            **kwargs)

    def quant_output_scale(self):
        return self.act_quant.scale()


class QuantNonLinearActMixin(QuantActMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            act_impl: Module,
            act_quant: Union[ActQuantProxy, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        QuantActMixin.__init__(
            self,
            act_impl=act_impl,
            act_quant=act_quant,
            update_injector=update_injector,
            proxy_impl=ActQuantProxy,
            proxy_prefix='act_',
            kwargs_prefix='',
            **kwargs)

    def quant_act_scale(self):
        return self.act_quant.scale()

