# global
import sys
import numpy as np

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = np.ndarray
NativeVariable = np.ndarray
Device = str
Dtype = np.dtype

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
from . import activations  # noqa
from .activations import *  # noqa
from . import creation  # noqa
from .creation import *  # noqa
from . import data_type  # noqa
from .data_type import *  # noqa
from . import device  # noqa
from .device import *  # noqa
from . import elementwise  # noqa
from .elementwise import *  # noqa
from . import general  # noqa
from .general import *  # noqa
from . import gradients  # noqa
from .gradients import *  # noqa
from . import image  # noqa
from .image import *  # noqa
from . import layers  # noqa
from .layers import *  # noqa
from . import linear_algebra as linalg  # noqa
from .linear_algebra import *  # noqa
from . import manipulation  # noqa
from .manipulation import *  # noqa
from . import random  # noqa
from .random import *  # noqa
from . import searching  # noqa
from .searching import *  # noqa
from . import set  # noqa
from .set import *  # noqa
from . import sorting  # noqa
from .sorting import *  # noqa
from . import statistical  # noqa
from .statistical import *  # noqa
from . import utility  # noqa
from .utility import *  # noqa
