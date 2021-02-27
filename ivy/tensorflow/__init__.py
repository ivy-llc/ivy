import sys
import ivy
from tensorflow.python.types.core import Tensor

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

Tensor = Tensor
backend = 'tensorflow'
