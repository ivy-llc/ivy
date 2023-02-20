# global
import sys
import paddle as paddle

# local
import ivy

backend_version = {"version": paddle.version.full_version}

# noinspection PyUnresolvedReferences
if not ivy.is_local():
    _module_in_memory = sys.modules[__name__]
else:
    _module_in_memory = sys.modules[ivy.import_module_path].import_cache[__name__]

use = ivy.utils.backend.ContextManager(_module_in_memory)

NativeArray = paddle.Tensor
NativeVariable = paddle.static.Variable  # paddle.fluid.framework.Variable
NativeDevice = paddle.fluid.libpaddle.Place
NativeDtype = paddle.dtype
NativeShape = list

NativeSparseArray = paddle.Tensor

# devices
valid_devices = ("cpu",)

invalid_devices = ("gpu", "tpu")


# native data types
native_int8 = paddle.int8
native_int16 = paddle.int16
native_int32 = paddle.int32
native_int64 = paddle.int64
native_uint8 = paddle.uint8
native_bfloat16 = paddle.bfloat16
native_float16 = paddle.float16
native_float32 = paddle.float32
native_float64 = paddle.float64
native_complex64 = paddle.complex64
native_complex128 = paddle.complex128
native_double = native_float64
native_bool = paddle.bool

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
    ivy.bfloat16,
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
)
valid_float_dtypes = (ivy.bfloat16, ivy.float16, ivy.float32, ivy.float64)
valid_uint_dtypes = (ivy.uint8,)
valid_complex_dtypes = (ivy.complex64, ivy.complex128)

invalid_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)
invalid_numeric_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)
invalid_int_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)
invalid_float_dtypes = ()
invalid_uint_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)
invalid_complex_dtypes = ()

native_inplace_support = True
supports_gradients = True


def closest_valid_dtype(type, /):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.as_ivy_dtype(type)
    if type_str in invalid_dtypes:
        return {"uint16": native_uint8, "uint32": native_uint8, "uint64": native_uint8}[
            type_str
        ]
    return type


backend = "paddle"

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
