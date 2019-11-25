import builtins
from torch.nn.parallel import DistributedDataParallel
from .apex_lightning import LightningApexDistributedDataParallel


# Ugly workaround

def _isinstance(instance, clz):
    if clz is DistributedDataParallel and builtins.isinstance_orig(instance, LightningApexDistributedDataParallel):
        return True
    return builtins.isinstance_orig(instance, clz)

builtins.isinstance_orig = builtins.isinstance
builtins.isinstance = _isinstance


from .model_lighting import QuantImageNetClassification


