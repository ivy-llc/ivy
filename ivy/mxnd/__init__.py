import sys
import ivy
import mxnet as mx

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

Array = mx.ndarray.ndarray.NDArray
Variable = mx.ndarray.ndarray.NDArray
Device = mx.context.Context
Dtype = type

backend = 'mxnd'
