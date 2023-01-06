import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase

__all__ = ["disable_running_stats", "enable_running_stats"]

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _NormBase):
            module.old_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _NormBase) and hasattr(module, "old_momentum"):
            module.momentum = module.old_momentum
            del module.old_momentum

    model.apply(_enable)