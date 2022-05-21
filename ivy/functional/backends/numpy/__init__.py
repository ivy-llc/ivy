# global
import sys
import numpy as np

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = np.ndarray
NativeVariable = np.ndarray
NativeDevice = str
NativeDtype = np.dtype

# data types
int8 = np.dtype("int8")
int16 = np.dtype("int16")
int32 = np.dtype("int32")
int64 = np.dtype("int64")
uint8 = np.dtype("uint8")
uint16 = np.dtype("uint16")
uint32 = np.dtype("uint32")
uint64 = np.dtype("uint64")
float16 = np.dtype("float16")
float32 = np.dtype("float32")
float64 = np.dtype("float64")
# noinspection PyShadowingBuiltins
bool = np.dtype("bool")

valid_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
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
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
)
valid_int_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
valid_float_dtypes = (float16, float32, float64)

# valid
valid_dtype_strs = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
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
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
)
valid_int_dtype_strs = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)
valid_float_dtype_strs = ("float16", "float32", "float64")

# invalid
invalid_dtype_strs = ("bfloat16",)
invalid_numeric_dtype_strs = ("bfloat16",)
invalid_int_dtype_strs = ()
invalid_float_dtype_strs = ("bfloat16",)


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.dtype_to_str(type)
    if type_str in invalid_dtype_strs:
        return {"bfloat16": float16}[type_str]
    return type


backend = "numpy"


# local sub-modules
from . import activations
from .activations import *
from . import compilation
from .compilation import *
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
