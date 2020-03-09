import torch
import math

from brevitas.nn import QuantConv2d, QuantLinear


class MaxL2ScalingReg:

    def __init__(self, coeff):
        self.coeff = coeff

    def loss(self, model):
        loss = 0.0
        for name, mod in model.named_modules():
            if isinstance(mod, (QuantConv2d, QuantLinear)):
                abs_weight = mod.weight.view(mod.weight.size(0), -1).abs()
                max_per_channel = torch.max(abs_weight, dim=1)[0]
                max_per_tensor = torch.max(max_per_channel)
                l2_per_tensor = torch.norm(max_per_channel, p=2)
                scaled_l2_per_tensor = l2_per_tensor / math.sqrt(max_per_channel.view(-1).shape[0])
                loss += max_per_tensor - scaled_l2_per_tensor
        loss *= self.coeff
        return loss