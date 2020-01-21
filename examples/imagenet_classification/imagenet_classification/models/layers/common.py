import torch.nn.functional as F

from brevitas.quant_tensor import pack_quant_tensor


def multisample_dropout_classify(x, classifier, samples, rate, training):
    x, scale, bit_width = x
    x = x.view(x.size(0), -1)
    if training and samples == 0:
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
