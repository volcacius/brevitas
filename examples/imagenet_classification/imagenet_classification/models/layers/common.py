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

import torch
import torch.nn.functional as F

from brevitas.quant_tensor import pack_quant_tensor


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
