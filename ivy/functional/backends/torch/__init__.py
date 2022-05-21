# global
import sys
import torch as torch

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = torch.Tensor
NativeVariable = torch.Tensor
NativeDevice = torch.device
NativeDtype = torch.dtype

# data types
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
uint8 = torch.uint8
bfloat16 = torch.bfloat16
float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
# noinspection PyShadowingBuiltins
bool = torch.bool

valid_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    bfloat16,
    float16,
    float32,
    float64,
    bool,
)
valid_numeric_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    bfloat16,
    float16,
    float32,
    float64,
)
valid_int_dtypes = (int8, int16, int32, int64, uint8)
valid_float_dtypes = (bfloat16, float16, float32, float64)

# valid
valid_dtype_strs = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "bool",
)
valid_numeric_dtype_strs = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bfloat16",
    "float16",
    "float32",
    "float64",
)
valid_int_dtype_strs = ("int8", "int16", "int32", "int64", "uint8")
valid_float_dtype_strs = ("bfloat16", "float16", "float32", "float64")

# invalid
invalid_dtype_strs = ("uint16", "uint32", "uint64")
invalid_num_dtype_strs = ("uint16", "uint32", "uint64")
invalid_int_dtype_strs = ("uint16", "uint32", "uint64")
invalid_float_dtype_strs = ()


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.dtype_to_str(type)
    if type_str in invalid_dtype_strs:
        return {"uint16": uint8, "uint32": uint8, "uint64": uint8}[type_str]
    return type


backend = "torch"


# local sub-modules
from . import activations
from .activations import *
from . import compilation
from .compilation import *
from . import converters
from .converters import *
from . import creation
from .creation import *
from . import data_type
from .data_type import *
from . import device
from .device import *
from . import elementwise
from .elementwise import *
from . import general
from .general import *
from . import gradients
from .gradients import *
from . import image
from .image import *
from . import layers
from .layers import *
from . import linear_algebra as linalg
from .linear_algebra import *
from . import manipulation
from .manipulation import *
from . import random
from .random import *
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
