# global
import functools
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
    "2.4.0 and below": (
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
    "2.4.1 and above": (
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
    "2.4.0 and below": (
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
    "2.4.1 and above": (
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
    "2.5.1 and below": (
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
    )
}
valid_float_dtypes = {
    "2.4.0 and below": (ivy.float16, ivy.float32, ivy.float64),
    "2.4.1 and above": (ivy.bfloat16, ivy.float16, ivy.float32, ivy.float64),
}
valid_uint_dtypes = {"2.5.1 and below": (ivy.uint8,)}
valid_complex_dtypes = {"2.5.1 and below": (ivy.complex64, ivy.complex128)}

# leave these untouched
valid_dtypes = _dtype_from_version(valid_dtypes, backend_version)
valid_numeric_dtypes = _dtype_from_version(valid_numeric_dtypes, backend_version)
valid_int_dtypes = _dtype_from_version(valid_int_dtypes, backend_version)
valid_float_dtypes = _dtype_from_version(valid_float_dtypes, backend_version)
valid_uint_dtypes = _dtype_from_version(valid_uint_dtypes, backend_version)
valid_complex_dtypes = _dtype_from_version(valid_complex_dtypes, backend_version)


# update these to add new dtypes
invalid_dtypes = {
    "2.4.0 and below": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.bfloat16,
    ),
    "2.4.1 and above": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
    ),
}

invalid_numeric_dtypes = {
    "2.4.0 and below": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.bfloat16,
    ),
    "2.4.1 and above": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
    ),
}

invalid_int_dtypes = {"2.5.1 and below": (ivy.uint16, ivy.uint32, ivy.uint64)}
invalid_float_dtypes = {"2.4.0 and below": (ivy.bfloat16,), "2.4.1 and above": ()}
invalid_uint_dtypes = {"2.5.1 and below": (ivy.uint16, ivy.uint32, ivy.uint64)}
invalid_complex_dtypes = {"2.5.1 and below": ()}

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


def _get_dtype_max(dtype):
    dtype = str(dtype)
    dtype = 'uint8' if dtype == 'bool' else dtype
    if "float" in dtype or "complex" in dtype:
        return paddle_backend.finfo(dtype).max
    else:
        return paddle_backend.iinfo(dtype).max


def to_dtype_and_back(
        fn: Callable,
        supported_dtypes: Sequence[str] = ('int32', 'int64', 'float32', 'float64'),
) -> Callable:
    @functools.wraps(fn)
    def _to_dtype_and_back(*args, **kwargs):
        """
        To be used on native paddle functions that do not have Kernels for all the
        required dtypes. Casts the fn arguments to a supported dtype and then the fn
        result back to the required dtype.
        ......

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        supported_dtypes
            The list of the supported dtypes.

        Returns
        -------
            The return of the function if the current
            backend matches the argument backend.
            If not, it raises an InvalidBackendException
        """

        if not ivy.is_array(args[0]):
            assert ivy.is_array(args[0]), "`to_and_back` should be used on fn calls " \
                                          "that have an array as their first argument"

        arr_dtype = str(ivy.as_ivy_dtype(args[0].dtype))

        if arr_dtype in supported_dtypes:
            return fn(*args, **kwargs)

        dtypes = sorted(list(supported_dtypes), key=_get_dtype_max)

        for dtype in dtypes:
            if _get_dtype_max(arr_dtype) <= _get_dtype_max(dtype):
                mid_dtype = dtype
                break

        if 'dtype' in kwargs:
            ret_dtype = kwargs['dtype']
            kwargs['dtype'] = mid_dtype
        else:
            ret_dtype = arr_dtype

        if 'complex' in arr_dtype:
            ret = paddle.complex(
                fn(args[0].real(), *args[1:], **kwargs),
                fn(args[0].imag(), *args[1:], **kwargs)
            )
        else:
            ret = fn(args[0].cast(mid_dtype), *args[1:], **kwargs)

        return ret.cast(ret_dtype)

    _to_dtype_and_back.to_dtype_and_back = True
    return _to_dtype_and_back
