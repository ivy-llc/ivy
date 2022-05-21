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
int8 = ivy.IntDtype("int8")
int16 = ivy.IntDtype("int16")
int32 = ivy.IntDtype("int32")
int64 = ivy.IntDtype("int64")
uint8 = ivy.IntDtype("uint8")
uint16 = ivy.IntDtype("uint16")
uint32 = ivy.IntDtype("uint32")
uint64 = ivy.IntDtype("uint64")
bfloat16 = ivy.FloatDtype("bfloat16")
float16 = ivy.FloatDtype("float16")
float32 = ivy.FloatDtype("float32")
float64 = ivy.FloatDtype("float64")
# noinspection PyShadowingBuiltins
bool = "bool"
nan = float("nan")
inf = float("inf")

# native data types
native_int8 = torch.int8
native_int16 = torch.int16
native_int32 = torch.int32
native_int64 = torch.int64
native_uint8 = torch.uint8
native_bfloat16 = torch.bfloat16
native_float16 = torch.float16
native_float32 = torch.float32
native_float64 = torch.float64
# noinspection PyShadowingBuiltins
native_bool = torch.bool

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

# invalid
invalid_dtypes = (uint16, uint32, uint64)
invalid_num_dtypes = (uint16, uint32, uint64)
invalid_int_dtypes = (uint16, uint32, uint64)
invalid_float_dtypes = ()


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.as_ivy_dtype(type)
    if type_str in invalid_dtypes:
        return {"uint16": native_uint8, "uint32": native_uint8, "uint64": native_uint8}[
            type_str
        ]
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
