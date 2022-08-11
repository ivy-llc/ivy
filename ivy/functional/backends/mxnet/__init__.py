# global
import sys
import mxnet as mx
import numpy as np

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.backend_handler.ContextManager(sys.modules[__name__])

NativeArray = mx.nd.NDArray
NativeVariable = mx.nd.NDArray
NativeDevice = mx.context.Context
NativeDtype = type
NativeShape = tuple

# native data types
native_int8 = np.int8
native_int32 = np.int32
native_int64 = np.int64
native_uint8 = np.uint8
native_float16 = np.float16
native_float32 = np.float32
native_float64 = np.float64
# noinspection PyShadowingBuiltins
native_bool = np.bool

# valid data types
valid_dtypes = (
    ivy.int8,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
    ivy.float16,
    ivy.float32,
    ivy.float64,
    ivy.bool,
)
valid_numeric_dtypes = (
    ivy.int8,
    ivy.int32,
    ivy.int64,
    ivy.uint8,
    ivy.float16,
    ivy.float32,
    ivy.float64,
)
valid_int_dtypes = (ivy.int8, ivy.int32, ivy.int64, ivy.uint8)
valid_float_dtypes = (ivy.float16, ivy.float32, ivy.float64)
valid_uint_dtypes = (ivy.uint8,)

# invalid data types
invalid_dtypes = (ivy.int16, ivy.uint16, ivy.uint32, ivy.uint64, ivy.bfloat16)
invalid_numeric_dtypes = (ivy.int16, ivy.uint16, ivy.uint32, ivy.uint64, ivy.bfloat16)
invalid_int_dtypes = (ivy.int16, ivy.uint16, ivy.uint32, ivy.uint64)
invalid_float_dtypes = (ivy.bfloat16,)
invalid_uint_dtypes = (ivy.uint16, ivy.uint32, ivy.uint64)

supports_gradients = True


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.as_ivy_dtype(type)
    if type_str in invalid_dtypes:
        return {
            "int16": ivy.int32,
            "uint16": ivy.uint8,
            "uint32": ivy.uint8,
            "uint64": ivy.uint8,
            "bfloat16": ivy.float16,
        }[type_str]
    return type


backend = "mxnet"

# Helpers #
# --------#


def _raise(ex):
    raise ex


def _mxnet_init_context(device):  # noqa
    device = ivy.as_ivy_dev(device)
    if device is None or device.find("cpu") != -1:
        mx_dev = "cpu"
    elif device.find("gpu") != -1:
        mx_dev = "gpu"
    else:
        raise Exception("dev input {} not supported.".format(device))
    if device.find(":") != -1:
        mx_dev_id = int(device[device.find(":") + 1 :])
    else:
        mx_dev_id = 0
    return mx.Context(mx_dev, mx_dev_id)


def _scalar_or_flat_array_to_scalar(x):
    return (
        x
        if isinstance(x, Number)  # noqa
        else (x.asscalar() if len(x.shape) == 0 else x)
    )


def _flat_array_to_1_dim_array(x):
    return (
        mx.nd.array([x.asscalar()]).astype(dtype(x)) if len(x.shape) == 0 else x  # noqa
    )


def _1_dim_array_to_flat_array(x):
    return mx.nd.array(x.asscalar(), dtype=x.dtype) if x.shape == (1,) else x


def _handle_flat_arrays_in(fn):
    return _handle_flat_arrays_in_out(fn, False)


def _handle_flat_arrays_in_out(fn, include_out=True):
    def wrapped_fn(*args, **kwargs):
        expanded = False

        def expand(x):
            nonlocal expanded
            expanded = True
            return _flat_array_to_1_dim_array(x)

        args_expanded = ivy.nested_map(
            args,
            lambda x: expand(x) if ivy.is_native_array(x) and len(x.shape) == 0 else x,
        )
        kwargs_expanded = ivy.nested_map(
            kwargs,
            lambda x: expand(x) if ivy.is_native_array(x) and len(x.shape) == 0 else x,
        )
        ret = fn(*args_expanded, **kwargs_expanded)
        if expanded and include_out:
            return ivy.nested_map(
                ret,
                lambda x: _1_dim_array_to_flat_array(x)
                if ivy.is_native_array(x)
                else x,
            )
        return ret

    return wrapped_fn


def _handle_output(x, axis, keepdims, ret):
    if not keepdims and (
        axis is None or len((axis,) if isinstance(axis, int) else axis) == len(x.shape)
    ):
        return _1_dim_array_to_flat_array(ret)
    return ret


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
