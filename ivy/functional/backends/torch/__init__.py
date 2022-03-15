# global
import sys
import torch as _torch

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = _torch.Tensor
NativeVariable = _torch.Tensor
Device = _torch.device
Dtype = _torch.dtype

# data types
int8 = _torch.int8
int16 = _torch.int16
int32 = _torch.int32
int64 = _torch.int64
uint8 = _torch.uint8
bfloat16 = _torch.bfloat16
float16 = _torch.float16
float32 = _torch.float32
float64 = _torch.float64
# noinspection PyShadowingBuiltins
bool = _torch.bool

all_dtypes = (int8, int16, int32, int64,
              uint8,
              bfloat16, float16, float32, float64)
valid_dtypes = all_dtypes

all_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                  'uint8',
                  'bfloat16', 'float16', 'float32', 'float64')
valid_dtype_strs = all_dtypes
invalid_dtype_strs = ('uint16', 'uint32', 'uint64')


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = dtype_to_str(type)
    if type_str in invalid_dtype_strs:
        return {'uint16': uint8,
                'uint32': uint8,
                'uint64': uint8}[type_str]
    return type


backend = 'torch'


# local sub-modules
from . import activations
from .activations import *
from . import constants
from .constants import *
from . import converters
from .converters import *
from . import creation
from .creation import *
from . import data_type
from .data_type import *
from . import elementwise
from .elementwise import *
from . import layers
from .layers import *
from . import linear_algebra as linalg
from .linear_algebra import *
from . import manipulation
from .manipulation import *
from . import searching
from .searching import *
from . import set
from .set import *
from . import sorting
from .sorting import *
from . import statistical
from .statistical import *
from . import utility
from .utility import *

from . import old
from .old import *
