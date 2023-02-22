# global
import sys
import mindspore as ms

# local
import ivy

backend_version = {"version": ms.__version__}

# noinspection PyUnresolvedReferences
if not ivy.is_local():
    _module_in_memory = sys.modules[__name__]
else:
    _module_in_memory = sys.modules[ivy.import_module_path].import_cache[__name__]

use = ivy.utils.backend.ContextManager(_module_in_memory)

NativeArray = ms.Tensor
NativeVariable = ms.Tensor
NativeDevice = ms.device
NativeDtype = ms.dtype
NativeShape = ms.Size

NativeSparseArray = ms.Tensor


# devices
valid_devices = ("cpu", "gpu")

invalid_devices = ("tpu",)


# native data types
native_int8 = ms.int8
native_int16 = ms.int16
native_int32 = ms.int32
native_int64 = ms.int64
native_uint8 = ms.uint8
native_float16 = ms.float16
native_float32 = ms.float32
native_float64 = ms.float64
native_complex64 = ms.complex64
native_complex128 = ms.complex128
native_double = ms.double
native_bool = ms.bool_

# valid data types
# ToDo: Add complex dtypes to valid_dtypes and fix all resulting failures.
valid_dtypes = (
    ivy.int8,
    ivy.int16,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
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
    ivy.bfloat16,
)
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


backend = "mindspore"

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
