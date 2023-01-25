# global
import sys
import numpy as np

# local
import ivy

backend_version = {"version": np.__version__}

# noinspection PyUnresolvedReferences
use = ivy.backend_handler.ContextManager(sys.modules[__name__])

NativeArray = np.ndarray
NativeVariable = np.ndarray
NativeDevice = str
NativeDtype = np.dtype
NativeShape = tuple

NativeSparseArray = None


# devices
valid_devices = ("cpu",)

invalid_devices = ("gpu", "tpu")

# native data types
native_int8 = np.dtype("int8")
native_int16 = np.dtype("int16")
native_int32 = np.dtype("int32")
native_int64 = np.dtype("int64")
native_uint8 = np.dtype("uint8")
native_uint16 = np.dtype("uint16")
native_uint32 = np.dtype("uint32")
native_uint64 = np.dtype("uint64")
native_float16 = np.dtype("float16")
native_float32 = np.dtype("float32")
native_float64 = np.dtype("float64")
native_complex64 = np.dtype("complex64")
native_complex128 = np.dtype("complex128")
native_double = native_float64
native_bool = np.dtype("bool")

# valid data types
# ToDo: Add complex dtypes to valid_dtypes and fix all resulting failures.
valid_dtypes = (
    ivy.int8,
    ivy.int16,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
    ivy.uint16,
    ivy.uint32,
    ivy.uint64,
    ivy.float16,
    ivy.float32,
    ivy.float64,
    ivy.complex64,
    ivy.complex128,
    ivy.bool,
)
valid_numeric_dtypes = (
    ivy.int8,
    ivy.int16,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
    ivy.uint16,
    ivy.uint32,
    ivy.uint64,
    ivy.float16,
    ivy.float32,
    ivy.float64,
)
valid_int_dtypes = (
    ivy.int8,
    ivy.int16,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
    ivy.uint16,
    ivy.uint32,
    ivy.uint64,
)
valid_float_dtypes = (ivy.float16, ivy.float32, ivy.float64)
valid_uint_dtypes = (ivy.uint8, ivy.uint16, ivy.uint32, ivy.uint64)
valid_complex_dtypes = (ivy.complex64, ivy.complex128)

# invalid data types
invalid_dtypes = (ivy.bfloat16,)
invalid_numeric_dtypes = (ivy.bfloat16,)
invalid_int_dtypes = ()
invalid_float_dtypes = (ivy.bfloat16,)
invalid_uint_dtypes = ()
invalid_complex_dtypes = ()

native_inplace_support = False

supports_gradients = False


def closest_valid_dtype(type, /):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.as_ivy_dtype(type)
    if type_str in invalid_dtypes:
        return {"bfloat16": ivy.float16}[type_str]
    return type


backend = "numpy"


# local sub-modules
from . import activations
from .activations import *
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
from . import experimental
from .experimental import *
from . import control_flow_ops
from .control_flow_ops import *
