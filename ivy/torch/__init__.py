import sys
import ivy

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

Array = torch.Tensor
Variable = torch.Tensor


def tensor_classes() -> List[torch.Tensor]:
    return [torch.Tensor]


backend = 'torch'
