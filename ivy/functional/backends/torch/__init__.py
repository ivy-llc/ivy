# global
import sys

import torch as torch

# local
import ivy

backend_version = {"version": torch.__version__.split("+")[0]}

# noinspection PyUnresolvedReferences
use = ivy.backend_handler.ContextManager(sys.modules[__name__])

NativeArray = torch.Tensor
NativeVariable = torch.Tensor
NativeDevice = torch.device
NativeDtype = torch.dtype
NativeShape = torch.Size

NativeSparseArray = torch.Tensor


# devices
valid_devices = ("cpu",)

invalid_devices = ("gpu", "tpu")


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
native_complex64 = torch.complex64
native_complex128 = torch.complex128
native_double = native_float64
native_bool = torch.bool

# valid data types
# ToDo: Add complex dtypes to valid_dtypes and fix all resulting failures.
valid_dtypes = (
    ivy.int8,
    ivy.int16,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
    ivy.bfloat16,
    ivy.float16,
    ivy.float32,
    ivy.float64,
    ivy.bool,
)
valid_numeric_dtypes = (
    ivy.int8,
    ivy.int16,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
    ivy.bfloat16,
    ivy.float16,
    ivy.float32,
    ivy.float64,
)
valid_int_dtypes = (ivy.int8, ivy.int16, ivy.int32, ivy.int64, ivy.uint8)
valid_float_dtypes = (ivy.bfloat16, ivy.float16, ivy.float32, ivy.float64)
valid_uint_dtypes = (ivy.uint8,)
valid_complex_dtypes = (ivy.complex64, ivy.complex128)

# invalid data types
invalid_dtypes = (
    ivy.uint16,
    ivy.uint32,
    ivy.uint64,
)
invalid_num_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)
invalid_int_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)
invalid_float_dtypes = ()
invalid_uint_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)
invalid_complex_dtypes = (ivy.complex256,)

native_inplace_support = True

supports_gradients = True


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
from . import (
    activations,
    compilation,
    converters,
    creation,
    data_type,
    device,
    elementwise,
    extensions,
    general,
    gradients,
    layers,
)
from . import linear_algebra as linalg
from . import manipulation, random, searching, set, sorting, statistical, utility
from .activations import *
from .compilation import *
from .converters import *
from .creation import *
from .data_type import *
from .device import *
from .elementwise import *
from .extensions import *
from .general import *
from .gradients import *
from .layers import *
from .linear_algebra import *
from .manipulation import *
from .random import *
from .searching import *
from .set import *
from .sorting import *
from .statistical import *
from .utility import *
