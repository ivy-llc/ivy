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
Device = mx.context.Context
Dtype = type

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
    type_str = ivy.dtype_to_str(type)
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
        x if isinstance(x, Number) else (x.asscalar() if len(x.shape) == 0 else x)  # noqa
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
