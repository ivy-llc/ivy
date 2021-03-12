import sys
import ivy
from tensorflow.python.types.core import Tensor
from tensorflow.python.framework.dtypes import DType

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

Array = Tensor
Variable = Tensor
Device = str
Dtype = DType

backend = 'tensorflow'
