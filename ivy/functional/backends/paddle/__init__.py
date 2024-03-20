# global
import sys
import paddle as paddle

# local
import ivy
from ivy.func_wrapper import _dtype_from_version

backend_version = {"version": paddle.version.full_version}

# noinspection PyUnresolvedReferences
if not ivy.is_local():
    _module_in_memory = sys.modules[__name__]
else:
    _module_in_memory = sys.modules[ivy.import_module_path].import_cache[__name__]

use = ivy.utils.backend.ContextManager(_module_in_memory)


# wrap dunder methods of native tensors to return NotImplemented to prioritize Ivy array methods.
def dunder_wrapper(func):
    def rep_method(*args, **kwargs):
        for arg in args:
            if ivy.is_ivy_array(arg):
                return NotImplemented
        return func(*args, **kwargs)

    return rep_method


# check for previously imported paddle modules
modules_to_patch = []
tensors_to_patch = []
tmp_globals = dict(globals())
for name, value in tmp_globals.items():
    if value == "paddle.Tensor":
        tensors_to_patch.append(name)
    try:
        if value.__name__ == "paddle":
            modules_to_patch.append(name)
    except AttributeError:
        pass

methods_to_patch = [
    "__add__",
    "__sub__",
    "__mul__",
    "__div__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__ne__",
    "__eq__",
    "__and__",
    "__or__",
    "__xor__",
    "__pow__",
    "__matmul__",
]

for module in modules_to_patch:
    for method in methods_to_patch:
        exec(
            module
            + ".Tensor."
            + method
            + " = dunder_wrapper("
            + module
            + ".Tensor."
            + method
            + ")"
        )

for tensor in tensors_to_patch:
    for method in methods_to_patch:
        exec(tensor + "." + method + " = dunder_wrapper(" + tensor + "." + method + ")")


NativeArray = paddle.Tensor
NativeVariable = paddle.Tensor  # paddle.fluid.framework.Variable
NativeDevice = paddle.device.core.Place
NativeDtype = paddle.dtype
NativeShape = list

NativeSparseArray = paddle.Tensor

# devices
valid_devices = (
    "cpu",
    "gpu",
)

invalid_devices = "tpu"


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

# update these to add new dtypes
valid_dtypes = {
    "2.4.2 and below": (
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
    ),
    "2.5.0 and above": (
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
    ),
}
valid_numeric_dtypes = {
    "2.4.2 and below": (
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
    ),
    "2.5.0 and above": (
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
    ),
}
valid_int_dtypes = {
    "2.6.0 and below": (
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
    ),
}
valid_float_dtypes = {
    "2.4.2 and below": (ivy.float16, ivy.float32, ivy.float64),
    "2.5.0 and above": (ivy.bfloat16, ivy.float16, ivy.float32, ivy.float64),
}
valid_uint_dtypes = {"2.6.0 and below": (ivy.uint8,)}
valid_complex_dtypes = {"2.6.0 and below": (ivy.complex64, ivy.complex128)}

# leave these untouched
valid_dtypes = _dtype_from_version(valid_dtypes, backend_version)
valid_numeric_dtypes = _dtype_from_version(valid_numeric_dtypes, backend_version)
valid_int_dtypes = _dtype_from_version(valid_int_dtypes, backend_version)
valid_float_dtypes = _dtype_from_version(valid_float_dtypes, backend_version)
valid_uint_dtypes = _dtype_from_version(valid_uint_dtypes, backend_version)
valid_complex_dtypes = _dtype_from_version(valid_complex_dtypes, backend_version)


# update these to add new dtypes
invalid_dtypes = {
    "2.4.2 and below": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.bfloat16,
    ),
    "2.5.0 and above": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
    ),
}

invalid_numeric_dtypes = {
    "2.4.2 and below": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.bfloat16,
    ),
    "2.5.0 and above": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
    ),
}

invalid_int_dtypes = {"2.6.0 and below": (ivy.uint16, ivy.uint32, ivy.uint64)}
invalid_float_dtypes = {"2.4.2 and below": (ivy.bfloat16,), "2.5.0 and above": ()}
invalid_uint_dtypes = {"2.6.0 and below": (ivy.uint16, ivy.uint32, ivy.uint64)}
invalid_complex_dtypes = {"2.6.0 and below": ()}

# leave these untouched
invalid_dtypes = _dtype_from_version(invalid_dtypes, backend_version)
invalid_numeric_dtypes = _dtype_from_version(invalid_numeric_dtypes, backend_version)
invalid_float_dtypes = _dtype_from_version(invalid_float_dtypes, backend_version)
invalid_uint_dtypes = _dtype_from_version(invalid_uint_dtypes, backend_version)
invalid_complex_dtypes = _dtype_from_version(invalid_complex_dtypes, backend_version)


native_inplace_support = False
supports_gradients = True


def closest_valid_dtype(type=None, /, as_native=False):
    if type is None:
        return ivy.default_dtype()
    if isinstance(type, str) and type in invalid_dtypes:
        type = {
            "uint16": native_uint8,
            "uint32": native_uint8,
            "uint64": native_uint8,
            "bfloat16": native_float16,
        }[type]
    return ivy.as_ivy_dtype(type) if not as_native else ivy.as_native_dtype(type)


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
from . import module
from .module import *


# sub-backends

from . import sub_backends
from .sub_backends import *


NativeModule = paddle.nn.Layer
