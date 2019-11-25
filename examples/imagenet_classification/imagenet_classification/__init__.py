import builtins
from pytorch_lightning.pt_overrides.override_data_parallel import LightningDistributedDataParallel
from .apex_lightning import LightningApexDistributedDataParallel


# Ugly workaround

def _isinstance(instance, clz):
    if clz is LightningDistributedDataParallel \
            and builtins.isinstance_orig(instance, LightningApexDistributedDataParallel):
        return True
    return builtins.isinstance_orig(instance, clz)

builtins.isinstance_orig = builtins.isinstance
builtins.isinstance = _isinstance


from .model_lighting import QuantImageNetClassification


