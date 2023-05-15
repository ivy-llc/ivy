import sys
import numpy as np
import mxnet as mx
import ivy
from ivy.func_wrapper import _dtype_from_version
from ivy.utils.exceptions import IvyNotImplementedException

backend_version = {"version": mx.__version__}
if not ivy.is_local():
    _module_in_memory = sys.modules[__name__]
else:
    _module_in_memory = sys.modules[ivy.import_module_path].import_cache[__name__]
use = ivy.utils.backend.ContextManager(_module_in_memory)


NativeArray = mx.ndarray.NDArray
NativeDevice = str
NativeDtype = np.dtype
NativeShape = tuple
NativeSparseArray = mx.ndarray.sparse.BaseSparseNDArray


valid_devices = ("cpu", "gpu")
invalid_devices = ("tpu",)

# native data types
native_int8 = np.dtype("int8")
native_int32 = np.dtype("int32")
native_int64 = np.dtype("int64")
native_uint8 = np.dtype("uint8")
native_float16 = np.dtype("float16")
native_float32 = np.dtype("float32")
native_float64 = np.dtype("float64")
native_double = native_float64
native_bool = np.dtype("bool")


valid_dtypes_dict = {
    "1.9.1 and below": (
        ivy.int8,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
        ivy.bfloat16,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.bool,
    )
}
valid_dtypes = _dtype_from_version(valid_dtypes_dict, backend_version)
valid_numeric_dtypes_dict = {
    "1.9.1 and below": (
        ivy.int8,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
        ivy.float16,
        ivy.float32,
        ivy.float64,
    )
}
valid_numeric_dtypes = _dtype_from_version(valid_numeric_dtypes_dict, backend_version)
valid_int_dtypes_dict = {"1.9.1 and below": (ivy.int8, ivy.int32, ivy.int64, ivy.uint8)}
valid_int_dtypes = _dtype_from_version(valid_int_dtypes_dict, backend_version)
valid_float_dtypes_dict = {"1.9.1 and below": (ivy.float16, ivy.float32, ivy.float64)}
valid_float_dtypes = _dtype_from_version(valid_float_dtypes_dict, backend_version)
valid_uint_dtypes_dict = {"1.9.1 and below": (ivy.uint8,)}
valid_uint_dtypes = _dtype_from_version(valid_uint_dtypes_dict, backend_version)
valid_complex_dtypes_dict = {"1.9.1 and below": ()}
valid_complex_dtypes = _dtype_from_version(valid_complex_dtypes_dict, backend_version)
invalid_dtypes_dict = {
    "1.9.1 and below": (ivy.int16, ivy.uint32, ivy.uint64, ivy.uint16)
}
invalid_dtypes = _dtype_from_version(invalid_dtypes_dict, backend_version)
invalid_numeric_dtypes_dict = {
    "1.9.1 and below": (ivy.int16, ivy.uint32, ivy.uint64, ivy.uint16)
}
invalid_numeric_dtypes = _dtype_from_version(
    invalid_numeric_dtypes_dict, backend_version
)
invalid_int_dtypes_dict = {
    "1.9.1 and below": (ivy.int16, ivy.uint16, ivy.uint32, ivy.uint64)
}
invalid_int_dtypes = _dtype_from_version(invalid_int_dtypes_dict, backend_version)
invalid_float_dtypes_dict = {"1.9.1 and below": (ivy.bfloat16,)}
invalid_float_dtypes = _dtype_from_version(invalid_float_dtypes_dict, backend_version)
invalid_uint_dtypes_dict = {"1.9.1 and below": (ivy.uint16, ivy.uint32, ivy.uint64)}
invalid_uint_dtypes = _dtype_from_version(invalid_uint_dtypes_dict, backend_version)
invalid_complex_dtypes_dict = {"1.9.1 and below": (ivy.complex64, ivy.complex128)}
invalid_complex_dtypes = _dtype_from_version(
    invalid_complex_dtypes_dict, backend_version
)
native_inplace_support = True
supports_gradients = True


def closest_valid_dtype(type=None, /, as_native=False):
    raise IvyNotImplementedException()


backend = "mxnet"
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
from . import sub_backends
from .sub_backends import *
