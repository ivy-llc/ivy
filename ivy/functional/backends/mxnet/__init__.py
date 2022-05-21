# global
import sys
import mxnet as mx
import numpy as np

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = mx.ndarray.ndarray.NDArray
NativeVariable = mx.ndarray.ndarray.NDArray
NativeDevice = mx.context.Context
NativeDtype = type

# data types
int8 = np.int8
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
float16 = np.float16
float32 = np.float32
float64 = np.float64
# noinspection PyShadowingBuiltins
bool = np.bool

valid_dtypes = (int8, int32, int64, uint8, float16, float32, float64, bool)
valid_numeric_dtypes = (int8, int32, int64, uint8, float16, float32, float64)
valid_int_dtypes = (int8, int32, int64, uint8)
valid_float_dtypes = (float16, float32, float64)

# valid
valid_dtype_strs = (
    "int8",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float32",
    "float64",
    "bool",
)
valid_numeric_dtype_strs = (
    "int8",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float32",
    "float64",
)
valid_int_dtype_strs = ("int8", "int32", "int64", "uint8")
valid_float_dtype_strs = ("float16", "float32", "float64")

# invalid
invalid_dtype_strs = ("int16", "uint16", "uint32", "uint64", "bfloat16")
invalid_numeric_dtype_strs = ("int16", "uint16", "uint32", "uint64", "bfloat16")
invalid_int_dtype_strs = ("int16", "uint16", "uint32", "uint64")
invalid_float_dtype_strs = ("bfloat16",)


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.as_ivy_dtype(type)
    if type_str in invalid_dtype_strs:
        return {
            "int16": int32,
            "uint16": uint8,
            "uint32": uint8,
            "uint64": uint8,
            "bfloat16": float16,
        }[type_str]
    return type


backend = "mxnet"

# Helpers #
# --------#


def _raise(ex):
    raise ex


def _mxnet_init_context(device):  # noqa
    device = ivy.dev_to_str(device)
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
