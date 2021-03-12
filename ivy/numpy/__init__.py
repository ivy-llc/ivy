import sys
import ivy
import numpy as np

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

Array = np.ndarray
Variable = np.ndarray
Device = str
Dtype = np.dtype

backend = 'numpy'
