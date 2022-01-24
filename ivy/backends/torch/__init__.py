import sys
import ivy
import torch as _torch

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = _torch.Tensor
NativeVariable = _torch.Tensor
Device = _torch.device
Dtype = _torch.dtype

backend = 'torch'
