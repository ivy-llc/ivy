import sys
import ivy
import mxnet as mx

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

Tensor = mx.ndarray.ndarray.NDArray
backend = 'mxnd'
