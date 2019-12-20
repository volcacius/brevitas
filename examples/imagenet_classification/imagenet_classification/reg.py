import torch

from brevitas.nn import QuantConv2d, QuantLinear


class MaxAveScalingReg:

    def __init__(self, coeff):
        self.coeff = coeff

    def loss(self, model):
        loss = 0.0
        for name, mod in model.named_modules():
            if isinstance(mod, (QuantConv2d, QuantLinear)):
                weight = mod.weight.view(mod.weight.size(0), -1)
                max_per_channel = torch.max(weight, dim=1)[0]
                loss += max_per_channel.std(unbiased=False)
        loss *= self.coeff
        return loss