from builtins import map as _map
from collections import UserDict
from numbers import Number
from numpy.core.numeric import normalize_axis_tuple
from operator import mul
from .tensorflow_NestedSequence_bknd import tensorflow_NestedSequence_bknd
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union
import ast
import copy
import functools
import inspect
import itertools
import math
import numpy as np
import os
import re
import tensorflow
import tensorflow as tf


promotion_table = {
    ("bool", "bool"): "bool",
    ("int8", "int8"): "int8",
    ("int8", "int16"): "int16",
    ("int8", "int32"): "int32",
    ("int8", "int64"): "int64",
    ("int16", "int16"): "int16",
    ("int16", "int32"): "int32",
    ("int16", "int64"): "int64",
    ("int32", "int32"): "int32",
    ("int32", "int64"): "int64",
    ("int64", "int64"): "int64",
    ("uint8", "int8"): "int16",
    ("uint8", "int16"): "int16",
    ("uint8", "int32"): "int32",
    ("uint8", "int64"): "int64",
    ("uint8", "uint8"): "uint8",
    ("uint8", "uint16"): "uint16",
    ("uint8", "uint32"): "uint32",
    ("uint8", "uint64"): "uint64",
    ("uint16", "int8"): "int32",
    ("uint16", "int16"): "int32",
    ("uint16", "int32"): "int32",
    ("uint16", "int64"): "int64",
    ("uint16", "uint16"): "uint16",
    ("uint16", "uint32"): "uint32",
    ("uint16", "uint64"): "uint64",
    ("uint32", "int8"): "int64",
    ("uint32", "int16"): "int64",
    ("uint32", "int32"): "int64",
    ("uint32", "int64"): "int64",
    ("uint32", "uint32"): "uint32",
    ("uint32", "uint64"): "uint64",
    ("uint64", "uint64"): "uint64",
    ("float16", "float16"): "float16",
    ("float16", "float32"): "float32",
    ("float16", "float64"): "float64",
    ("float32", "float32"): "float32",
    ("float32", "float64"): "float64",
    ("float64", "float64"): "float64",
    ("bool", "int8"): "int8",
    ("bool", "int16"): "int16",
    ("bool", "int32"): "int32",
    ("bool", "int64"): "int64",
    ("bool", "uint8"): "uint8",
    ("bool", "uint16"): "uint16",
    ("bool", "uint32"): "uint32",
    ("bool", "uint64"): "uint64",
    ("bool", "float16"): "float16",
    ("bool", "float32"): "float32",
    ("bool", "float64"): "float64",
    ("bool", "bfloat16"): "bfloat16",
    ("bool", "complex64"): "complex64",
    ("bool", "complex128"): "complex128",
    ("int8", "float16"): "float16",
    ("int8", "float32"): "float32",
    ("int8", "float64"): "float64",
    ("int8", "bfloat16"): "bfloat16",
    ("int8", "complex64"): "complex64",
    ("int8", "complex128"): "complex128",
    ("int16", "float32"): "float32",
    ("int16", "float64"): "float64",
    ("int16", "complex64"): "complex64",
    ("int16", "complex128"): "complex128",
    ("int32", "float64"): "float64",
    ("int32", "complex128"): "complex128",
    ("int64", "float64"): "float64",
    ("int64", "complex128"): "complex128",
    ("uint8", "float16"): "float16",
    ("uint8", "float32"): "float32",
    ("uint8", "float64"): "float64",
    ("uint8", "bfloat16"): "bfloat16",
    ("uint8", "complex64"): "complex64",
    ("uint8", "complex128"): "complex128",
    ("uint16", "float32"): "float32",
    ("uint16", "float64"): "float64",
    ("uint16", "complex64"): "complex64",
    ("uint16", "complex128"): "complex128",
    ("uint32", "float64"): "float64",
    ("uint32", "complex128"): "complex128",
    ("uint64", "int8"): "float64",
    ("uint64", "int16"): "float64",
    ("uint64", "int32"): "float64",
    ("uint64", "int64"): "float64",
    ("uint64", "float64"): "float64",
    ("uint64", "complex128"): "complex128",
    ("float16", "bfloat16"): "float32",
    ("float16", "complex64"): "complex64",
    ("float16", "complex128"): "complex128",
    ("float32", "complex64"): "complex64",
    ("float32", "complex128"): "complex128",
    ("float64", "complex64"): "complex128",
    ("float64", "complex128"): "complex128",
    ("bfloat16", "float16"): "float32",
    ("bfloat16", "float32"): "float32",
    ("bfloat16", "float64"): "float64",
    ("bfloat16", "bfloat16"): "bfloat16",
    ("bfloat16", "complex64"): "complex64",
    ("bfloat16", "complex128"): "complex128",
    ("complex64", "float64"): "complex128",
    ("complex64", "complex64"): "complex64",
    ("complex64", "complex128"): "complex128",
    ("complex128", "complex128"): "complex128",
    ("float16", "int16"): "float32",
    ("float16", "int32"): "float64",
    ("float16", "int64"): "float64",
    ("float16", "uint16"): "float32",
    ("float16", "uint32"): "float64",
    ("float16", "uint64"): "float64",
    ("float32", "int32"): "float64",
    ("float32", "int64"): "float64",
    ("float32", "uint32"): "float64",
    ("float32", "uint64"): "float64",
    ("bfloat16", "int16"): "float32",
    ("bfloat16", "int32"): "float64",
    ("bfloat16", "int64"): "float64",
    ("bfloat16", "uint16"): "float32",
    ("bfloat16", "uint32"): "float64",
    ("bfloat16", "uint64"): "float64",
    ("complex64", "int32"): "complex128",
    ("complex64", "int64"): "complex128",
    ("complex64", "uint32"): "complex128",
    ("complex64", "uint64"): "complex128",
}
array_api_promotion_table = {
    ("bool", "bool"): "bool",
    ("int8", "int8"): "int8",
    ("int8", "int16"): "int16",
    ("int8", "int32"): "int32",
    ("int8", "int64"): "int64",
    ("int16", "int16"): "int16",
    ("int16", "int32"): "int32",
    ("int16", "int64"): "int64",
    ("int32", "int32"): "int32",
    ("int32", "int64"): "int64",
    ("int64", "int64"): "int64",
    ("uint8", "int8"): "int16",
    ("uint8", "int16"): "int16",
    ("uint8", "int32"): "int32",
    ("uint8", "int64"): "int64",
    ("uint8", "uint8"): "uint8",
    ("uint8", "uint16"): "uint16",
    ("uint8", "uint32"): "uint32",
    ("uint8", "uint64"): "uint64",
    ("uint16", "int8"): "int32",
    ("uint16", "int16"): "int32",
    ("uint16", "int32"): "int32",
    ("uint16", "int64"): "int64",
    ("uint16", "uint16"): "uint16",
    ("uint16", "uint32"): "uint32",
    ("uint16", "uint64"): "uint64",
    ("uint32", "int8"): "int64",
    ("uint32", "int16"): "int64",
    ("uint32", "int32"): "int64",
    ("uint32", "int64"): "int64",
    ("uint32", "uint32"): "uint32",
    ("uint32", "uint64"): "uint64",
    ("uint64", "uint64"): "uint64",
    ("float16", "float16"): "float16",
    ("float16", "float32"): "float32",
    ("float16", "float64"): "float64",
    ("float32", "float32"): "float32",
    ("float32", "float64"): "float64",
    ("float64", "float64"): "float64",
}
tf.experimental.numpy.experimental_enable_numpy_behavior(True)
default_device_stack = []
SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")
default_uint_dtype_stack = []
default_complex_dtype_stack = []
default_dtype_stack = []
default_float_dtype_stack = []
ivy_dtype_dict = {
    tensorflow.int8: "int8",
    tensorflow.int16: "int16",
    tensorflow.int32: "int32",
    tensorflow.int64: "int64",
    tensorflow.uint8: "uint8",
    tensorflow.uint16: "uint16",
    tensorflow.uint32: "uint32",
    tensorflow.uint64: "uint64",
    tensorflow.bfloat16: "bfloat16",
    tensorflow.float16: "float16",
    tensorflow.float32: "float32",
    tensorflow.float64: "float64",
    tensorflow.complex64: "complex64",
    tensorflow.complex128: "complex128",
    tensorflow.bool: "bool",
}
default_int_dtype_stack = []
backend = ""
native_dtype_dict = {
    "int8": tensorflow.int8,
    "int16": tensorflow.int16,
    "int32": tensorflow.int32,
    "int64": tensorflow.int64,
    "uint8": tensorflow.uint8,
    "uint16": tensorflow.uint16,
    "uint32": tensorflow.uint32,
    "uint64": tensorflow.uint64,
    "bfloat16": tensorflow.bfloat16,
    "float16": tensorflow.float16,
    "float32": tensorflow.float32,
    "float64": tensorflow.float64,
    "complex64": tensorflow.complex64,
    "complex128": tensorflow.complex128,
    "bool": tensorflow.bool,
}
CONV_FUNCS = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
NORM_FUNCS = [
    "_BatchNorm",
    "_InstanceNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "SyncBatchNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LocalResponseNorm",
]
POOL_FUNCS = [
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "FractionalMaxPool2d",
    "LPPool1d",
    "LPPool2d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
]
KERAS_CONV_FUNCS = [
    "KerasConv1D",
    "KerasConv2D",
    "KerasConv3D",
    "KerasDepthwiseConv2D",
    "KerasConv1DTranspose",
    "KerasConv2DTranspose",
    "KerasConv3DTranspose",
]
KERAS_NORM_FUNCS = [
    "KerasBatchNorm1D",
    "KerasBatchNorm2D",
    "KerasBatchNorm3D",
    "KerasLayerNormalization",
    "KerasGroupNormalization",
    "KerasUnitNorm1D",
    "KerasUnitNorm2D",
    "KerasUnitNorm3D",
]
KERAS_POOL_FUNCS = [
    "KerasAveragePooling1D",
    "KerasAveragePooling2D",
    "KerasAveragePooling3D",
    "KerasMaxPool1D",
    "KerasMaxPool2D",
    "KerasMaxPool3D",
]
PADDING_FUNCS = [
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad2d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
]
KERAS_PADDING_FUNCS = ["KerasZeroPadding1D", "KerasZeroPadding2D", "KerasZeroPadding3D"]
ACTIVATION_FUNCS = [
    "ELU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "LeakyReLU",
    "PReLU",
    "ReLU",
    "ReLU6",
    "RReLU",
    "SELU",
    "CELU",
    "GELU",
    "Sigmoid",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
    "AdaptiveLogSoftmaxWithLoss",
]
KERAS_ACTIVATION_FUNCS = [
    "KerasReLU",
    "KerasPReLU",
    "KerasLeakyReLU",
    "KerasThresholdedReLU",
    "KerasELU",
    "KerasSoftmax",
]
DROPOUT_FUNCS = [
    "Dropout",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
]
KERAS_DROPOUT_FUNCS = ["KerasDropout"]
CONV_BLOCK_FNS = [
    *CONV_FUNCS,
    *KERAS_CONV_FUNCS,
    *POOL_FUNCS,
    *KERAS_POOL_FUNCS,
    *PADDING_FUNCS,
    *KERAS_PADDING_FUNCS,
    *ACTIVATION_FUNCS,
    *KERAS_ACTIVATION_FUNCS,
    *NORM_FUNCS,
    *KERAS_NORM_FUNCS,
    *DROPOUT_FUNCS,
    *KERAS_DROPOUT_FUNCS,
]
DATA_FORMAT = "PT"


def tensorflow_handle_array_like_without_promotion(fn: Callable):
    @functools.wraps(fn)
    def _handle_array_like_without_promotion(*args, **kwargs):
        args = list(args)
        num_args = len(args)
        try:
            type_hints = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        parameters = list(type_hints.keys())
        annotations = [param.annotation for param in type_hints.values()]
        device = tensorflow__get_preferred_device(args, kwargs)
        for i, (annotation, parameter, arg) in enumerate(
            zip(annotations, parameters, args)
        ):
            annotation_str = str(annotation)
            if (
                ("rray" in annotation_str or "Tensor" in annotation_str)
                and parameter != "out"
                and all(
                    sq not in annotation_str
                    for sq in ["Sequence", "List", "Tuple", "float", "int", "bool"]
                )
            ):
                if i < num_args:
                    if tensorflow__check_in_nested_sequence(
                        arg, value=Ellipsis, _type=slice
                    ):
                        continue
                    if not tensorflow_is_array_bknd(arg):
                        args = tensorflow_set_item_bknd(
                            args, i, tensorflow_asarray(arg, device=device)
                        )
                elif parameters in kwargs:
                    kwarg = tensorflow_get_item(kwargs, parameter)
                    if not tensorflow_is_array_bknd(kwarg):
                        kwargs = tensorflow_set_item_bknd(
                            kwargs, parameter, tensorflow_asarray(kwarg, device=device)
                        )
        return fn(*args, **kwargs)

    _handle_array_like_without_promotion.handle_array_like_without_promotion = True
    return _handle_array_like_without_promotion


def tensorflow_store_config_info(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if all(
            [
                hasattr(self, "_args"),
                hasattr(self, "_kwargs"),
                hasattr(self, "_self_tracked_trackables"),
            ]
        ):
            orig_trackables = copy.copy(self._self_tracked_trackables)
            self._args = (self,) + args
            self._kwargs = kwargs
            self._self_tracked_trackables = orig_trackables

    return wrapper


def tensorflow_is_native_array(x, /, *, exclusive=False):
    if "keras.src.backend.tensorflow.core.Variable" in str(x.__class__):
        return not exclusive
    if isinstance(x, (tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray)):
        if exclusive and isinstance(x, tensorflow.Variable):
            return False
        return True
    return False


def tensorflow_is_ivy_array_bknd(
    x: Union[tensorflow.Tensor, tf.Tensor], /, *, exclusive: Optional[bool] = False
):
    return isinstance(x, tensorflow.Tensor) and tensorflow_is_native_array(
        x, exclusive=exclusive
    )


def tensorflow_is_array_bknd(x: Any, /, *, exclusive: bool = False):
    return tensorflow_is_ivy_array_bknd(
        x, exclusive=exclusive
    ) or tensorflow_is_native_array(x, exclusive=exclusive)


def tensorflow_handle_methods(fn):
    def extract_function_name(s):
        match = re.search("_(.+?)(?:_\\d+)?$", s)
        if match:
            return match.group(1)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if tensorflow_is_array_bknd(args[0]):
            return fn(*args, **kwargs)
        else:
            pattern = "_bknd_|_bknd|_frnt_|_frnt"
            fn_name = extract_function_name(re.sub(pattern, "", fn.__name__))
            new_fn = getattr(args[0], fn_name)
            return new_fn(*args[1:], **kwargs)

    return wrapper


def tensorflow_exists_bknd(x: Any, /):
    return x is not None


def tensorflow_default_bknd(
    x: Any,
    /,
    default_val: Any,
    *,
    catch_exceptions: bool = False,
    rev: bool = False,
    with_callable: bool = False,
):
    with_callable = catch_exceptions or with_callable
    if rev:
        x, default_val = default_val, x
    if with_callable:
        x_callable = callable(x)
        default_callable = callable(default_val)
    else:
        x_callable = False
        default_callable = False
    if catch_exceptions:
        try:
            x = x() if x_callable else x
        except Exception:
            return default_val() if default_callable else default_val
    else:
        x = x() if x_callable else x
    return (
        x
        if tensorflow_exists_bknd(x)
        else default_val() if default_callable else default_val
    )


def tensorflow_nested_argwhere_bknd(
    nest: Iterable,
    fn: Callable,
    check_nests: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    _index: Optional[List] = None,
    _base: bool = True,
    stop_after_n_found: Optional[int] = None,
):
    to_ignore = tensorflow_default_bknd(to_ignore, ())
    _index = [] if _index is None else _index
    if isinstance(nest, (tuple, list)) and not isinstance(nest, to_ignore):
        n = 0
        _indices = []
        for i, item in enumerate(nest):
            ind = (
                tensorflow_nested_argwhere_bknd(
                    item,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [i],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else tensorflow_nested_argwhere_bknd(
                    item, fn, check_nests, to_ignore, _index + [i], False, None
                )
            )
            if stop_after_n_found is not None and ind:
                if n >= stop_after_n_found:
                    break
                n = n + len(ind)
            _indices = _indices + [ind]
            if stop_after_n_found is not None and n >= stop_after_n_found:
                break
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    elif isinstance(nest, (dict, UserDict)) and not isinstance(nest, to_ignore):
        n = 0
        _indices = []
        for k, v in nest.items():
            ind = (
                tensorflow_nested_argwhere_bknd(
                    v,
                    fn,
                    check_nests,
                    to_ignore,
                    _index + [k],
                    False,
                    stop_after_n_found - n,
                )
                if stop_after_n_found is not None
                else tensorflow_nested_argwhere_bknd(
                    v, fn, check_nests, to_ignore, _index + [k], False, None
                )
            )
            if stop_after_n_found is not None and ind:
                if n >= stop_after_n_found:
                    break
                n = n + len(ind)
            _indices = _indices + [ind]
        _indices = [idx for idxs in _indices if idxs for idx in idxs]
        if check_nests and fn(nest):
            _indices.append(_index)
    else:
        cond_met = fn(nest)
        if cond_met:
            return [_index]
        return False
    return [index for index in _indices if index]


def tensorflow__check_float64_bknd(input):
    if tensorflow_is_array_bknd(input):
        return tensorflow_dtype(input) == "float64"
    if math.isfinite(input):
        m, e = math.frexp(input)
        return abs(input) > 3.4028235e38 or e < -126 or e > 128
    return False


def tensorflow_as_ivy_dtype_bknd(dtype_in: Union[str, str], /):
    return tensorflow_as_ivy_dtype(dtype_in)


def tensorflow_is_complex_dtype_bknd(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array_bknd(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype_bknd()
    elif isinstance(dtype_in, np.ndarray):
        return "complex" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (complex, np.complexfloating))
    elif isinstance(dtype_in, (list, tuple, dict)):
        return tensorflow_nested_argwhere_bknd(
            dtype_in,
            lambda x: isinstance(x, (complex, np.complexfloating))
            or tensorflow_is_array_bknd(x)
            and "complex" in tensorflow_dtype(x),
        )
    return "complex" in tensorflow_as_ivy_dtype_bknd(dtype_in)


def tensorflow_index_nest_bknd(
    nest: Union[List, Tuple, Dict, tensorflow.Tensor, tf.Tensor, dict],
    index: Union[List[int], Tuple[int], Iterable[int]],
    /,
):
    ret = nest
    for i in index:
        ret = tensorflow_get_item(ret, i)
    return ret


def tensorflow__get_first_array(*args, **kwargs):
    def array_fn(x):
        return (
            tensorflow_is_array_bknd(x)
            if not hasattr(x, "_ivy_array")
            else tensorflow_is_array_bknd(x.ivy_array)
        )

    array_fn = array_fn if "array_fn" not in kwargs else kwargs["array_fn"]
    arr = None
    if args:
        arr_idxs = tensorflow_nested_argwhere_bknd(args, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = tensorflow_index_nest_bknd(args, arr_idxs[0])
        else:
            arr_idxs = tensorflow_nested_argwhere_bknd(
                kwargs, array_fn, stop_after_n_found=1
            )
            if arr_idxs:
                arr = tensorflow_index_nest_bknd(kwargs, arr_idxs[0])
    elif kwargs:
        arr_idxs = tensorflow_nested_argwhere_bknd(
            kwargs, array_fn, stop_after_n_found=1
        )
        if arr_idxs:
            arr = tensorflow_index_nest_bknd(kwargs, arr_idxs[0])
    return arr


def tensorflow_as_native_dev(device: str, /):
    if isinstance(device, str) and "/" in device:
        return device
    ret = f"/{str(device).upper()}"
    if not ret[-1].isnumeric():
        ret += ":0"
    return ret


def tensorflow_handle_methods_1(fn):
    def extract_function_name(s):
        match = re.search("_(.+?)(?:_\\d+)?$", s)
        if match:
            return match.group(1)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if tensorflow_is_array_bknd(args[0]):
            return fn(*args, **kwargs)
        else:
            pattern = "_bknd_|_bknd|_frnt_|_frnt"
            fn_name = extract_function_name(re.sub(pattern, "", fn.__name__))
            new_fn = getattr(args[0], fn_name)
            return new_fn(*args[1:], **kwargs)

    return wrapper


@tensorflow_handle_methods_1
def tensorflow_split(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[
        Union[int, Sequence[int], Union[tensorflow.Tensor, tensorflow.Variable]]
    ] = None,
    axis: int = 0,
    with_remainder: bool = False,
):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception(
                f"input array had no shape, but num_sections specified was {num_or_size_splits}"
            )
        return [x]
    if num_or_size_splits is None:
        dim_size = tensorflow.shape(x)[axis]
        num_or_size_splits = int(dim_size)
    if isinstance(num_or_size_splits, (tensorflow.Tensor, tensorflow.Variable)):
        num_or_size_splits = tensorflow.cast(num_or_size_splits, tensorflow.int32)
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [
                int(remainder * num_or_size_splits)
            ]
    return tensorflow.split(x, num_or_size_splits, axis)


@tensorflow_handle_methods_1
def tensorflow_split_bknd_(
    self: tensorflow.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[
        Union[int, Sequence[int], tensorflow.Tensor, tf.Tensor]
    ] = None,
    axis: int = 0,
    with_remainder: bool = False,
):
    return tensorflow_split(
        self,
        copy=copy,
        num_or_size_splits=num_or_size_splits,
        axis=axis,
        with_remainder=with_remainder,
    )


def tensorflow_as_ivy_dev(device: str, /):
    if isinstance(device, str) and "/" not in device:
        return str(device)
    dev_in_split = tensorflow_split_bknd_(device[1:], ":")[-2:]
    if len(dev_in_split) == 1:
        return str(dev_in_split[0])
    dev_type, dev_idx = dev_in_split[0], dev_in_split[1]
    dev_type = dev_type.lower()
    if dev_type == "cpu":
        return str(dev_type)
    return str(f"{dev_type}:{dev_idx}")


def tensorflow_stack(
    arrays: Union[Tuple[tensorflow.Tensor], List[tensorflow.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    try:
        return tensorflow.experimental.numpy.stack(arrays, axis)
    except ValueError as e:
        raise Exception(e) from e


def tensorflow_stack_bknd_(
    self: tensorflow.Tensor,
    /,
    arrays: Union[
        Tuple[Union[tensorflow.Tensor, tf.Tensor]],
        List[Union[tensorflow.Tensor, tf.Tensor]],
    ],
    *,
    axis: int = 0,
    out: Optional[tensorflow.Tensor] = None,
):
    if not isinstance(arrays, (tuple, list)):
        arrays = [arrays]
    if isinstance(arrays, tuple):
        x = (self,) + arrays
    else:
        x = [self] + arrays
    return tensorflow_stack(x, axis=axis, out=out)


def tensorflow_dev(
    x: Union[tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray],
    /,
    *,
    as_native: bool = False,
):
    if "keras.src.backend.tensorflow.core.Variable" in str(x.__class__):
        x = x.value
    if isinstance(x, tensorflow.TensorArray):
        x = tensorflow_stack_bknd_(x)
    dv = x.device
    if as_native:
        return dv
    dv = dv if dv else tensorflow_default_device_bknd(as_native=False)
    return tensorflow_as_ivy_dev(dv)


def tensorflow_default_device_bknd(
    device: Optional[Union[str, str]] = None,
    /,
    *,
    item: Optional[Union[list, tuple, dict, tensorflow.Tensor, tf.Tensor]] = None,
    as_native: Optional[bool] = None,
):
    if tensorflow_exists_bknd(device):
        if as_native is True:
            return tensorflow_as_native_dev(device)
        elif as_native is False:
            return tensorflow_as_ivy_dev(device)
        return device
    as_native = tensorflow_default_bknd(as_native, False)
    if tensorflow_exists_bknd(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif tensorflow_is_array_bknd(item):
            return tensorflow_dev(item, as_native=as_native)
    global default_device_stack
    if not default_device_stack:
        ret = "cpu"
    else:
        ret = default_device_stack[-1]
    if as_native:
        return tensorflow_as_native_dev(ret)
    return tensorflow_as_ivy_dev(ret)


def tensorflow__get_preferred_device(args, kwargs):
    device = None
    if "device" in kwargs and kwargs["device"] is not None:
        return device
    if not False:
        arr_arg = tensorflow__get_first_array(*args, **kwargs)
        return tensorflow_default_device_bknd(item=arr_arg, as_native=True)
    return tensorflow_default_device_bknd(as_native=True)


def tensorflow__check_in_nested_sequence(sequence, value=None, _type=None):
    if sequence is value or isinstance(sequence, _type):
        return True
    elif isinstance(sequence, (tuple, list)):
        if any(isinstance(_val, _type) or _val is value for _val in sequence):
            return True
        else:
            return any(
                tensorflow__check_in_nested_sequence(sub_sequence, value, _type)
                for sub_sequence in sequence
                if isinstance(sub_sequence, (tuple, list))
            )


def tensorflow_is_variable(x, /, *, exclusive=False):
    return isinstance(x, tensorflow.Variable)


def tensorflow_variable(x, /):
    with tensorflow.device(tensorflow_dev(x, as_native=True)):
        return tensorflow.Variable(x, trainable=True)


@tensorflow_handle_array_like_without_promotion
def tensorflow_stop_gradient(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    preserve_type: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    is_var = tensorflow_is_variable(x)
    x = tensorflow.stop_gradient(x)
    if is_var and preserve_type:
        return tensorflow_variable(x)
    return x


def tensorflow_nested_map_bknd(
    fn: Callable,
    x: Union[tensorflow.Tensor, tf.Tensor, Iterable],
    /,
    include_derived: Optional[Union[Dict[str, bool], bool]] = None,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    to_mutable: bool = False,
    _tuple_check_fn: Optional[Callable] = None,
    _list_check_fn: Optional[Callable] = None,
    _dict_check_fn: Optional[Callable] = None,
    shallow: bool = True,
):
    to_ignore = tensorflow_default_bknd(to_ignore, ())
    if include_derived is True:
        include_derived = {"tuple": True, "list": True, "dict": True}
    elif not include_derived:
        include_derived = {}
    for t in ("tuple", "list", "dict"):
        if t not in include_derived:
            include_derived = tensorflow_set_item_bknd(include_derived, t, False)
    class_instance = type(x)
    if (
        hasattr(x, "is_tracked_proxy")
        and hasattr(class_instance, "__bases__")
        and not set(class_instance.__bases__).intersection(set(to_ignore))
    ):
        to_ignore = to_ignore + (class_instance,)
    tuple_check_fn = tensorflow_default_bknd(
        _tuple_check_fn,
        (
            (lambda x_, t_: isinstance(x_, t_))
            if include_derived["tuple"]
            else lambda x_, t_: type(x_) is t_
        ),
    )
    list_check_fn = tensorflow_default_bknd(
        _list_check_fn,
        (
            (lambda x_, t_: isinstance(x_, t_))
            if include_derived["list"]
            else lambda x_, t_: type(x_) is t_
        ),
    )
    dict_check_fn = tensorflow_default_bknd(
        _dict_check_fn,
        (
            (lambda x_, t_: isinstance(x_, t_))
            if include_derived["dict"]
            else lambda x_, t_: type(x_) is t_
        ),
    )
    if tuple_check_fn(x, tuple) and not isinstance(x, to_ignore):
        ret_list = [
            tensorflow_nested_map_bknd(
                fn,
                i,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for i in x
        ]
        if to_mutable:
            return ret_list
        elif hasattr(x, "_fields"):
            return class_instance(**dict(zip(x._fields, ret_list)))
        else:
            return class_instance(ret_list)
    elif list_check_fn(x, list) and not isinstance(x, to_ignore):
        ret_list = [
            tensorflow_nested_map_bknd(
                fn,
                i,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for i in x
        ]
        if shallow:
            x = tensorflow_set_item_bknd(x, slice(None, None, None), ret_list[:])
            return x
        return class_instance(ret_list)
    elif (dict_check_fn(x, dict) or isinstance(x, UserDict)) and not isinstance(
        x, to_ignore
    ):
        class_instance = type(x)
        ret = {
            k: tensorflow_nested_map_bknd(
                fn,
                v,
                include_derived,
                to_ignore,
                to_mutable,
                tuple_check_fn,
                list_check_fn,
                dict_check_fn,
                shallow,
            )
            for k, v in x.items()
        }
        if shallow:
            x.update(ret)
            return x
        return class_instance(ret)
    elif isinstance(x, slice):
        return slice(*tensorflow_nested_map_bknd(fn, [x.start, x.stop, x.step]))
    return fn(x)


def tensorflow__to_ivy_bknd_(x: Any):
    if isinstance(x, tensorflow.Tensor):
        return x
    elif isinstance(x, tf.TensorShape):
        return tuple(x)
    elif isinstance(x, dict):
        return x.to_ivy()
    if tensorflow_is_native_array(x) or isinstance(x, np.ndarray):
        return tensorflow.convert_to_tensor(x)
    return x


def tensorflow_to_ivy_bknd_(
    x: Union[tensorflow.Tensor, tf.Tensor, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[str, bool]] = None,
):
    if nested:
        return tensorflow_nested_map_bknd(
            tensorflow__to_ivy_bknd_, x, include_derived, shallow=False
        )
    return tensorflow__to_ivy_bknd_(x)


def tensorflow__asarray_to_native_arrays_and_back_bknd(fn: Callable):
    @functools.wraps(fn)
    def _asarray_to_native_arrays_and_back_wrapper(*args, dtype=None, **kwargs):
        new_arg = args[0]
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = tensorflow_default_dtype_bknd(dtype=dtype, as_native=True)
        return tensorflow_to_ivy_bknd_(fn(*new_args, dtype=dtype, **kwargs))

    _asarray_to_native_arrays_and_back_wrapper._asarray_to_native_arrays_and_back = True
    return _asarray_to_native_arrays_and_back_wrapper


def tensorflow__flatten_nest_bknd(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from tensorflow__flatten_nest_bknd(x)
        else:
            yield x


def tensorflow_promote_types_bknd(
    type1: Union[str, tf.DType],
    type2: Union[str, tf.DType],
    /,
    *,
    array_api_promotion: bool = False,
):
    if not (type1 and type2):
        return type1 if type1 else type2
    query = [tensorflow_as_ivy_dtype(type1), tensorflow_as_ivy_dtype(type2)]
    query = tuple(query)
    if query not in promotion_table:
        query = query[1], query[0]

    def _promote(query):
        if array_api_promotion:
            return tensorflow_get_item(array_api_promotion_table, query)
        return tensorflow_get_item(promotion_table, query)

    return _promote(query)


def tensorflow__asarray_infer_dtype_bknd(fn: Callable):
    @functools.wraps(fn)
    def _asarray_infer_dtype_wrapper(*args, dtype=None, **kwargs):
        def _infer_dtype(obj):
            if isinstance(obj, tf.TensorShape):
                obj = list(obj)
            if hasattr(obj, "dtype"):
                return obj.dtype.name if isinstance(obj, np.ndarray) else obj.dtype
            else:
                return tensorflow_default_dtype_bknd(item=obj)

        if not tensorflow_exists_bknd(dtype):
            arr = args[0]
            dtype_list = [
                tensorflow_nested_map_bknd(
                    lambda x: _infer_dtype(x), arr, shallow=False
                )
            ]
            dtype_list = tensorflow__flatten_nest_bknd(dtype_list)
            dtype_list = list(set(dtype_list))
            if len(dtype_list) != 0:
                dtype = dtype_list[0]
                for dt in dtype_list[1:]:
                    dtype = tensorflow_promote_types_bknd(dtype, dt)
            else:
                dtype = tensorflow_default_float_dtype_bknd()
            dtype = tensorflow_as_native_dtype(dtype)
        return fn(*args, dtype=dtype, **kwargs)

    _asarray_infer_dtype_wrapper.infer_dtype = True
    return _asarray_infer_dtype_wrapper


@tensorflow_handle_array_like_without_promotion
@tensorflow__asarray_to_native_arrays_and_back_bknd
@tensorflow__asarray_infer_dtype_bknd
def tensorflow_asarray(
    obj: Union[
        tensorflow.Tensor,
        tensorflow.Variable,
        tensorflow.TensorShape,
        bool,
        int,
        float,
        tensorflow_NestedSequence_bknd,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[tensorflow.DType] = None,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    with tensorflow.device(device):
        if tensorflow.is_tensor(obj):
            ret = tensorflow.cast(obj, dtype) if obj.dtype != dtype else obj
        elif (
            dtype is not None
            and dtype.is_integer
            and np.issubdtype(np.array(obj).dtype, np.floating)
        ):
            obj_np = np.array(obj)
            ret = tensorflow.convert_to_tensor(obj_np, dtype)
        else:
            ret = tensorflow.convert_to_tensor(obj, dtype)
        return (
            tensorflow.identity(ret)
            if copy or tensorflow_as_native_dev(tensorflow_dev(ret)) != device
            else ret
        )


@tensorflow_handle_array_like_without_promotion
def tensorflow_size(x: tensorflow.Tensor, /):
    return functools.reduce(mul, x.shape) if len(x.shape) > 0 else 1


def tensorflow_size_bknd_(self):
    return tensorflow_size(self)


@tensorflow_handle_array_like_without_promotion
def tensorflow_unstack(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
):
    if x.shape == ():
        return [x]
    ret = tensorflow.unstack(x, axis=axis)
    if keepdims:
        return [tensorflow.expand_dims(r, axis) for r in ret]
    return ret


def tensorflow_unstack_bknd_(
    self: tensorflow.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
):
    return tensorflow_unstack(self, copy=copy, axis=axis, keepdims=keepdims)


@tensorflow_handle_array_like_without_promotion
def tensorflow_copy_array(
    x: Union[tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray],
    *,
    to_ivy_array: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if isinstance(x, tensorflow.TensorArray):
        x_wrapped = tensorflow_stack_bknd_(x)
        y = tensorflow.TensorArray(x.dtype, tensorflow_size_bknd_(x)())
        x = tensorflow_unstack_bknd_(y, tensorflow_copy_array(x_wrapped))
    else:
        x = tensorflow.identity(x)
    if to_ivy_array:
        return tensorflow_to_ivy_bknd_(x)
    return x


def tensorflow_tile(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    repeats: Sequence[int],
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if x.shape == ():
        x = tensorflow.reshape(x, (-1,))
    if isinstance(repeats, Number):
        repeats = [repeats]
    if isinstance(repeats, tensorflow.Tensor) and repeats.shape == ():
        repeats = tensorflow.reshape(repeats, (-1,))
    if len(x.shape) < len(repeats):
        while len(x.shape) != len(repeats):
            x = tensorflow.expand_dims(x, 0)
    elif len(x.shape) > len(repeats):
        repeats = list(repeats)
        while len(x.shape) != len(repeats):
            repeats = [1] + repeats
    return tensorflow.tile(x, repeats)


@tensorflow_handle_array_like_without_promotion
def tensorflow_nonzero(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
):
    res = tensorflow.experimental.numpy.nonzero(x)
    if size is not None:
        dtype = tensorflow.int64
        if isinstance(fill_value, float):
            dtype = tensorflow.float64
        res = tensorflow.cast(res, dtype)
        diff = size - res[0].shape[0]
        if diff > 0:
            res = tensorflow.pad(res, [[0, 0], [0, diff]], constant_values=fill_value)
        elif diff < 0:
            res = tensorflow.slice(res, [0, 0], [-1, size])
    if as_tuple:
        return tuple(res)
    return tensorflow.stack(res, axis=1)


@tensorflow_handle_array_like_without_promotion
def tensorflow_diff(
    x: Union[tensorflow.Tensor, tensorflow.Variable, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[
        Union[tensorflow.Tensor, tensorflow.Variable, int, float, list, tuple]
    ] = None,
    append: Optional[
        Union[tensorflow.Tensor, tensorflow.Variable, int, float, list, tuple]
    ] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if n == 0:
        return x
    if prepend is not None:
        x = tensorflow.experimental.numpy.append(
            prepend, x, axis=axis if axis != -1 else None
        )
    if append is not None:
        x = tensorflow.experimental.numpy.append(
            x, append, axis=axis if axis != -1 else None
        )
    return tensorflow.experimental.numpy.diff(x, n=n, axis=axis)


def tensorflow__parse_ellipsis_bknd(so, ndims):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    ret = list(
        pre
        + [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))]
        + list(reversed(post))
    )
    return ret, (len(pre), ndims - len(post))


def tensorflow_broadcast_arrays(*arrays: Union[tensorflow.Tensor, tensorflow.Variable]):
    if len(arrays) > 1:
        try:
            desired_shape = tensorflow.broadcast_dynamic_shape(
                tensorflow.shape(arrays[0]), tensorflow.shape(arrays[1])
            )
        except tensorflow.errors.InvalidArgumentError as e:
            raise Exception(e) from e
        if len(arrays) > 2:
            for i in range(2, len(arrays)):
                try:
                    desired_shape = tensorflow.broadcast_dynamic_shape(
                        desired_shape, tensorflow.shape(arrays[i])
                    )
                except tensorflow.errors.InvalidArgumentError as e:
                    raise Exception(e) from e
    else:
        return [arrays[0]]
    result = []
    for tensor in arrays:
        result.append(tensorflow.broadcast_to(tensor, desired_shape))
    return result


@tensorflow_handle_array_like_without_promotion
def tensorflow_astype(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    dtype: Union[tf.DType, str],
    /,
    *,
    copy: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dtype = tensorflow_as_native_dtype(dtype)
    if x.dtype == dtype:
        return tensorflow.experimental.numpy.copy(x) if copy else x
    return tensorflow.cast(x, dtype)


def tensorflow_astype_bknd_(
    self: tensorflow.Tensor,
    dtype: str,
    /,
    *,
    copy: bool = True,
    out: Optional[tensorflow.Tensor] = None,
):
    return tensorflow_astype(self, dtype, copy=copy, out=out)


@tensorflow_handle_array_like_without_promotion
def tensorflow_where(
    condition: Union[tensorflow.Tensor, tensorflow.Variable],
    x1: Union[tensorflow.Tensor, tensorflow.Variable],
    x2: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype if hasattr(x2, "dtype") else tensorflow_default_dtype_bknd()
        )
        if not tensorflow_is_array_bknd(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array_bknd(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    return tensorflow.cast(
        tensorflow.experimental.numpy.where(condition, x1, x2), x1.dtype
    )


@tensorflow_handle_array_like_without_promotion
def tensorflow_arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[tensorflow.DType] = None,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if stop is None:
        stop = start
        start = 0
    if step > 0 and start > stop or step < 0 and start < stop:
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start
    if isinstance(start, (float, int)):
        start = tensorflow.convert_to_tensor(start)
    if isinstance(stop, (float, int)):
        stop = tensorflow.convert_to_tensor(stop)
    if isinstance(step, (float, int)):
        step = tensorflow.convert_to_tensor(step)
    if dtype is None:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return tensorflow.cast(
                tensorflow.range(start, stop, delta=step, dtype=tensorflow.int64),
                tensorflow.int32,
            )
        else:
            return tensorflow.range(start, stop, delta=step)
    else:
        dtype = tensorflow_as_native_dtype(tensorflow_default_dtype_bknd(dtype=dtype))
        if dtype in [
            tensorflow.int8,
            tensorflow.uint8,
            tensorflow.int16,
            tensorflow.uint16,
            tensorflow.uint32,
            tensorflow.uint64,
        ]:
            return tensorflow.cast(
                tensorflow.range(start, stop, delta=step, dtype=tensorflow.int64), dtype
            )
        else:
            return tensorflow.range(start, stop, delta=step, dtype=dtype)


def tensorflow__parse_slice_bknd(idx, s):
    step = 1 if idx.step is None else idx.step
    if step > 0:
        start = 0 if idx.start is None else idx.start
        if start >= s:
            stop = start
        else:
            if start <= -s:
                start = 0
            elif start < 0:
                start = start + s
            stop = s if idx.stop is None else idx.stop
            if stop > s:
                stop = s
            elif start <= -s:
                stop = 0
            elif stop < 0:
                stop = stop + s
    else:
        start = s - 1 if idx.start is None else idx.start
        if start < -s:
            stop = start
        else:
            if start >= s:
                start = s - 1
            elif start < 0:
                start = start + s
            if idx.stop is None:
                stop = -1
            else:
                stop = idx.stop
                if stop > s:
                    stop = s
                elif stop < -s:
                    stop = -1
                elif stop == -s:
                    stop = 0
                elif stop < 0:
                    stop = stop + s
    q_i = tensorflow_arange(start, stop, step)
    ag__result_list_0 = []
    for q in q_i:
        if 0 <= q < s:
            res = q
            ag__result_list_0.append(res)
    q_i = ag__result_list_0
    q_i = (
        tensorflow_asarray(q_i)
        if len(q_i) or start == stop or idx.stop is not None
        else tensorflow_arange(0, s, 1)
    )
    return q_i


@tensorflow_handle_array_like_without_promotion
def tensorflow_shape(
    x: Union[tensorflow.Tensor, tensorflow.Variable], /, *, as_array: bool = False
):
    if as_array:
        return tensorflow_asarray(
            tensorflow.shape(x), dtype=tensorflow_default_int_dtype_bknd()
        )
    else:
        return tuple(x.shape)


def tensorflow__deep_flatten_bknd(iterable):
    def _flatten_gen(iterable):
        for item in iterable:
            if isinstance(item, list):
                yield from _flatten_gen(item)
            else:
                yield item

    return list(_flatten_gen(iterable))


def tensorflow__calculate_out_shape_bknd(axis, array_shape):
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_dims = len(axis) + len(array_shape)
    norm_axis = normalize_axis_tuple(axis, out_dims)
    shape_iter = iter(array_shape)
    ag__result_list_0 = []
    for current_ax in range(out_dims):
        res = 1 if current_ax in norm_axis else next(shape_iter)
        ag__result_list_0.append(res)
    out_shape = ag__result_list_0
    return out_shape


@tensorflow_handle_array_like_without_promotion
def tensorflow_expand_dims(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    try:
        out_shape = tensorflow__calculate_out_shape_bknd(axis, tensorflow.shape(x))
        ret = tensorflow.reshape(x, shape=out_shape)
        return ret
    except (tensorflow.errors.InvalidArgumentError, np.AxisError) as error:
        raise Exception(error) from error


def tensorflow_check_elem_in_list(elem, list, inverse=False, message=""):
    if inverse and elem in list:
        raise Exception(
            message if message != "" else f"{elem} must not be one of {list}"
        )
    elif not inverse and elem not in list:
        raise Exception(message if message != "" else f"{elem} must be one of {list}")


def tensorflow__reshape_fortran_tf(x, shape):
    if len(x.shape) > 0:
        x = tensorflow.transpose(x)
    return tensorflow.transpose(tensorflow.reshape(x, shape[::-1]))


@tensorflow_handle_array_like_without_promotion
def tensorflow_reshape(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            (new_s if con else old_s)
            for new_s, con, old_s in zip(
                shape, tensorflow.constant(shape) != 0, x.shape
            )
        ]
    if order == "F":
        return tensorflow__reshape_fortran_tf(x, shape)
    return tensorflow.reshape(x, shape)


def tensorflow_reshape_bknd_(
    self: tensorflow.Tensor,
    /,
    shape: Union[tuple, tf.TensorShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[tensorflow.Tensor] = None,
):
    return tensorflow_reshape(
        self, shape, copy=copy, allowzero=allowzero, out=out, order=order
    )


@tensorflow_handle_array_like_without_promotion
def tensorflow_meshgrid(
    *arrays: Union[tensorflow.Tensor, tensorflow.Variable],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if not sparse:
        return tensorflow.meshgrid(*arrays, indexing=indexing)
    sd = (1,) * len(arrays)
    ag__result_list_0 = []
    for i, a in enumerate(arrays):
        res = tensorflow.reshape(
            tensorflow.convert_to_tensor(a), sd[:i] + (-1,) + sd[i + 1 :]
        )
        ag__result_list_0.append(res)
    res = ag__result_list_0
    if indexing == "xy" and len(arrays) > 1:
        res[0] = tensorflow.reshape(res[0], (1, -1) + sd[2:])
        res[1] = tensorflow.reshape(res[1], (-1, 1) + sd[2:])
    return res


def tensorflow_infer_dtype(fn: Callable):
    @functools.wraps(fn)
    def _infer_dtype(*args, dtype=None, **kwargs):
        arr = (
            None
            if tensorflow_exists_bknd(dtype)
            else tensorflow__get_first_array(*args, **kwargs)
        )
        dtype = tensorflow_default_dtype_bknd(dtype=dtype, item=arr, as_native=True)
        return fn(*args, dtype=dtype, **kwargs)

    _infer_dtype.infer_dtype = True
    return _infer_dtype


@tensorflow_infer_dtype
@tensorflow_handle_array_like_without_promotion
def tensorflow_empty(
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    dtype: tensorflow.DType,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.experimental.numpy.empty(shape, dtype=tensorflow.float32)


def tensorflow__parse_query_bknd(query, x_shape, scatter=False):
    query = (query,) if not isinstance(query, tuple) else query
    ag__result_list_0 = []
    for q in query:
        res = tensorflow_asarray(q) if isinstance(q, (tuple, list, int)) else q
        ag__result_list_0.append(res)
    query = ag__result_list_0
    ag__result_list_1 = []
    for i, q in enumerate(query):
        if tensorflow_is_array_bknd(q):
            res = i
            ag__result_list_1.append(res)
    non_slice_q_idxs = ag__result_list_1
    to_front = (
        len(non_slice_q_idxs) > 1
        and any(tensorflow_diff(non_slice_q_idxs) != 1)
        and non_slice_q_idxs[-1] < len(x_shape)
    )
    ag__result_list_2 = []
    for i, q in enumerate(query):
        if q is None:
            res = i
            ag__result_list_2.append(res)
    new_axes = ag__result_list_2
    ag__result_list_3 = []
    for q in query:
        if q is not None:
            res = q
            ag__result_list_3.append(res)
    query = ag__result_list_3
    query = [Ellipsis] if query == [] else query
    ellipsis_inds = None
    if any(q is Ellipsis for q in query):
        query, ellipsis_inds = tensorflow__parse_ellipsis_bknd(query, len(x_shape))
    ag__result_list_4 = []
    for i, v in enumerate(query):
        if tensorflow_is_array_bknd(v):
            res = i
            ag__result_list_4.append(res)
    array_inds = ag__result_list_4
    if array_inds:
        array_queries = tensorflow_broadcast_arrays(
            *[v for i, v in enumerate(query) if i in array_inds]
        )
        array_queries = [
            (
                tensorflow_nonzero(q, as_tuple=False)[0]
                if tensorflow_is_bool_dtype_bknd(q)
                else q
            )
            for q in array_queries
        ]
        array_queries = [
            (
                tensorflow_astype_bknd_(
                    tensorflow_where(
                        arr < 0, arr + tensorflow_get_item(x_shape, i), arr
                    ),
                    tf.int64,
                )
                if tensorflow_size_bknd_(arr)
                else tensorflow_astype_bknd_(arr, tf.int64)
            )
            for arr, i in zip(array_queries, array_inds)
        ]
        for idx, arr in zip(array_inds, array_queries):
            query = tensorflow_set_item_bknd(query, idx, arr)
    ag__result_list_5 = []
    for i, q in enumerate(query):
        res = (
            tensorflow_astype_bknd_(
                tensorflow__parse_slice_bknd(q, tensorflow_get_item(x_shape, i)),
                tf.int64,
            )
            if isinstance(q, slice)
            else q
        )
        ag__result_list_5.append(res)
    query = ag__result_list_5
    if len(query) < len(x_shape):
        query = query + [
            tensorflow_astype_bknd_(tensorflow_arange(0, s, 1), tf.int64)
            for s in tensorflow_get_item(x_shape, slice(len(query), None, None))
        ]
    if len(array_inds) and to_front:
        target_shape = (
            [list(array_queries[0].shape)]
            + [
                list(tensorflow_get_item(query, i).shape)
                for i in range(len(query))
                if i not in array_inds
            ]
            + [[] for _ in range(len(array_inds) - 1)]
        )
    elif len(array_inds):
        target_shape = (
            [list(tensorflow_get_item(query, i).shape) for i in range(0, array_inds[0])]
            + [list(tensorflow_shape(array_queries[0], as_array=True))]
            + [[] for _ in range(len(array_inds) - 1)]
            + [
                list(tensorflow_shape(tensorflow_get_item(query, i), as_array=True))
                for i in range(array_inds[-1] + 1, len(query))
            ]
        )
    else:
        target_shape = [list(q.shape) for q in query]
    if ellipsis_inds is not None:
        target_shape = (
            tensorflow_get_item(target_shape, slice(None, ellipsis_inds[0], None))
            + [
                tensorflow_get_item(
                    target_shape, slice(ellipsis_inds[0], ellipsis_inds[1], None)
                )
            ]
            + tensorflow_get_item(target_shape, slice(ellipsis_inds[1], None, None))
        )
    for i, ax in enumerate(new_axes):
        if len(array_inds) and to_front:
            ax = ax - (sum(1 for x in array_inds if x < ax) - 1)
            ax = ax + i
        target_shape = [
            *tensorflow_get_item(target_shape, slice(None, ax, None)),
            1,
            *tensorflow_get_item(target_shape, slice(ax, None, None)),
        ]
    target_shape = tensorflow__deep_flatten_bknd(target_shape)
    ag__result_list_6 = []
    for q in query:
        res = tensorflow_expand_dims(q) if not len(q.shape) else q
        ag__result_list_6.append(res)
    query = ag__result_list_6
    if len(array_inds):
        array_queries = [
            (
                tensorflow_reshape_bknd_(arr, (-1,))
                if len(arr.shape) > 1
                else tensorflow_expand_dims(arr) if not len(arr.shape) else arr
            )
            for arr in array_queries
        ]
        array_queries = tensorflow_stack(array_queries, axis=1)
    if len(array_inds) == len(query):
        indices = tensorflow_reshape_bknd_(array_queries, (*target_shape, len(x_shape)))
    elif len(array_inds) == 0:
        indices = tensorflow_reshape_bknd_(
            tensorflow_stack(tensorflow_meshgrid(*query, indexing="ij"), axis=-1),
            (*target_shape, len(x_shape)),
        )
    elif to_front:
        post_array_queries = (
            tensorflow_reshape_bknd_(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i not in array_inds],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, len(query) - len(array_inds)),
            )
            if len(array_inds) < len(query)
            else tensorflow_empty((1, 0))
        )
        indices = tensorflow_reshape_bknd_(
            tensorflow_asarray(
                [
                    (*arr, *post)
                    for arr, post in itertools.product(
                        array_queries, post_array_queries
                    )
                ]
            ),
            (*target_shape, len(x_shape)),
        )
    else:
        pre_array_queries = (
            tensorflow_reshape_bknd_(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i < array_inds[0]],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, array_inds[0]),
            )
            if array_inds[0] > 0
            else tensorflow_empty((1, 0))
        )
        post_array_queries = (
            tensorflow_reshape_bknd_(
                tensorflow_stack(
                    tensorflow_meshgrid(
                        *[v for i, v in enumerate(query) if i > array_inds[-1]],
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                (-1, len(query) - 1 - array_inds[-1]),
            )
            if array_inds[-1] < len(query) - 1
            else tensorflow_empty((1, 0))
        )
        indices = tensorflow_reshape_bknd_(
            tensorflow_asarray(
                [
                    (*pre, *arr, *post)
                    for pre, arr, post in itertools.product(
                        pre_array_queries, array_queries, post_array_queries
                    )
                ]
            ),
            (*target_shape, len(x_shape)),
        )
    return (
        tensorflow_astype_bknd_(indices, tf.int64),
        target_shape,
        array_inds if len(array_inds) and to_front else None,
    )


def tensorflow_get_num_dims(x, /, *, as_array=False):
    return (
        tensorflow.cast(tensorflow.shape(tensorflow.shape(x))[0], tensorflow.int64)
        if as_array
        else int(tensorflow.shape(tensorflow.shape(x)))
    )


def tensorflow_to_numpy(
    x: Union[tensorflow.Tensor, tensorflow.Variable], /, *, copy: bool = True
):
    if (
        tensorflow_is_array_bknd(x)
        and tensorflow_get_num_dims(x) == 0
        and tensorflow_as_native_dtype(x.dtype) is tensorflow.bfloat16
    ):
        x = tensorflow.expand_dims(x, 0)
        if copy:
            return np.squeeze(np.array(tensorflow.convert_to_tensor(x)), 0)
        else:
            return np.squeeze(np.asarray(tensorflow.convert_to_tensor(x)), 0)
    if copy:
        return np.array(tensorflow.convert_to_tensor(x))
    else:
        return np.asarray(tensorflow.convert_to_tensor(x))


def tensorflow_to_scalar(x: Union[tensorflow.Tensor, tensorflow.Variable], /):
    ret = tensorflow_to_numpy(x).item()
    if x.dtype == tensorflow.bfloat16:
        return float(ret)
    return ret


@tensorflow_handle_array_like_without_promotion
def tensorflow_to_scalar_bknd(x: Union[tensorflow.Tensor, tf.Tensor], /):
    return tensorflow_to_scalar(x)


def tensorflow_is_float_dtype_bknd(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array_bknd(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype_bknd()
    elif isinstance(dtype_in, np.ndarray):
        return "float" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (float, np.floating))
    elif isinstance(dtype_in, (list, tuple, dict)):
        return bool(
            tensorflow_nested_argwhere_bknd(
                dtype_in,
                lambda x: isinstance(x, (float, np.floating))
                or tensorflow_is_array_bknd(x)
                and "float" in tensorflow_dtype(x),
            )
        )
    return "float" in tensorflow_as_ivy_dtype_bknd(dtype_in)


def tensorflow_is_uint_dtype_bknd(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array_bknd(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype_bknd()
    elif isinstance(dtype_in, np.ndarray):
        return "uint" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, np.unsignedinteger)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return tensorflow_nested_argwhere_bknd(
            dtype_in,
            lambda x: isinstance(x, np.unsignedinteger)
            or tensorflow_is_array_bknd(x)
            and "uint" in tensorflow_dtype(x),
        )
    return "uint" in tensorflow_as_ivy_dtype_bknd(dtype_in)


def tensorflow_default_uint_dtype_bknd(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    uint_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_uint_dtype_stack
    if tensorflow_exists_bknd(uint_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(uint_dtype)
        return str(tensorflow_as_ivy_dtype(uint_dtype))
    as_native = tensorflow_default_bknd(as_native, False)
    if tensorflow_exists_bknd(input):
        if tensorflow_is_array_bknd(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):

            def is_native(x):
                return tensorflow_is_native_array(x)

            if tensorflow_nested_argwhere_bknd(
                input,
                lambda x: (
                    tensorflow_dtype(x) == "uint64"
                    if is_native(x)
                    else x > 9223372036854775807 and x != math.inf
                ),
                stop_after_n_found=1,
            ):
                ret = tf.uint64
            elif default_uint_dtype_stack:
                ret = default_uint_dtype_stack[-1]
            else:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_uint_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "uint32"
        elif isinstance(input, Number):
            if input > 4294967295 and input != math.inf and backend != "torch":
                ret = tf.uint64
            elif default_uint_dtype_stack:
                ret = default_uint_dtype_stack[-1]
            else:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_uint_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "uint32"
    elif default_uint_dtype_stack:
        ret = default_uint_dtype_stack[-1]
    else:
        def_dtype = tensorflow_default_dtype_bknd()
        if tensorflow_is_uint_dtype_bknd(def_dtype):
            ret = def_dtype
        else:
            ret = "uint32"
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype(ret))


def tensorflow_is_int_dtype_bknd(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array_bknd(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, tuple):
        dtype_in = tensorflow_default_int_dtype_bknd()
    elif isinstance(dtype_in, np.ndarray):
        return "int" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (int, np.integer)) and not isinstance(
            dtype_in, bool
        )
    elif isinstance(dtype_in, (list, tuple, dict)):

        def nested_fun(x):
            return (
                isinstance(x, (int, np.integer))
                or tensorflow_is_array_bknd(x)
                and "int" in tensorflow_dtype(x)
            ) and x is not bool

        return bool(tensorflow_nested_argwhere_bknd(dtype_in, nested_fun))
    return "int" in tensorflow_as_ivy_dtype(dtype_in)


def tensorflow_infer_default_dtype_bknd(
    dtype: Union[str, tf.DType, str], as_native: bool = False
):
    if tensorflow_is_complex_dtype_bknd(dtype):
        default_dtype = tensorflow_default_complex_dtype_bknd(as_native=as_native)
    elif tensorflow_is_float_dtype_bknd(dtype):
        default_dtype = tensorflow_default_float_dtype_bknd(as_native=as_native)
    elif tensorflow_is_uint_dtype_bknd(dtype):
        default_dtype = tensorflow_default_uint_dtype_bknd(as_native=as_native)
    elif tensorflow_is_int_dtype_bknd(dtype):
        default_dtype = tensorflow_default_int_dtype_bknd(as_native=as_native)
    elif as_native:
        default_dtype = tensorflow_as_native_dtype("bool")
    else:
        default_dtype = tensorflow_as_ivy_dtype("bool")
    return default_dtype


def tensorflow_dtype_bits(dtype_in: Union[tensorflow.DType, str, np.dtype], /):
    dtype_str = tensorflow_as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("tf.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
        .replace("complex", "")
    )


def tensorflow__infer_dtype(dtype: tensorflow.DType):
    default_dtype = tensorflow_infer_default_dtype_bknd(dtype)
    if tensorflow_dtype_bits(dtype) < tensorflow_dtype_bits(default_dtype):
        return default_dtype
    return dtype


@tensorflow_handle_array_like_without_promotion
def tensorflow_prod(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[tensorflow.DType] = None,
    keepdims: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dtype = tensorflow_as_native_dtype(dtype)
    if dtype is None:
        dtype = tensorflow__infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tensorflow.experimental.numpy.prod(
        x, axis=axis, dtype=dtype, keepdims=keepdims
    )


def tensorflow__numel_bknd(shape):
    shape = tuple(shape)
    return tensorflow_to_scalar_bknd(tensorflow_prod(shape)) if shape != () else 1


def tensorflow_check_one_way_broadcastable(x1, x2):
    if len(x1) > len(x2):
        return False
    for a, b in zip(x1[::-1], x2[::-1]):
        if a in (1, b):
            pass
        else:
            return False
    return True


def tensorflow_check_shapes_broadcastable(var, data):
    if not tensorflow_check_one_way_broadcastable(var, data):
        raise Exception(f"Could not broadcast shape {data} to shape {var}.")


@tensorflow_handle_array_like_without_promotion
def tensorflow_broadcast_to(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_shapes_broadcastable(x.shape, shape)
    if tensorflow.rank(x) > len(shape):
        return tensorflow.broadcast_to(tensorflow.reshape(x, -1), shape)
    return tensorflow.broadcast_to(x, shape)


def tensorflow__broadcast_to_bknd(input, target_shape):
    if tensorflow__numel_bknd(tuple(input.shape)) == tensorflow__numel_bknd(
        tuple(target_shape)
    ):
        return tensorflow_reshape(input, target_shape)
    else:
        input = input if len(input.shape) else tensorflow_expand_dims(input, axis=0)
        return tensorflow_broadcast_to(input, target_shape)


@tensorflow_handle_array_like_without_promotion
def tensorflow_any(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    try:
        return tensorflow.reduce_any(
            tensorflow.cast(x, tensorflow.bool), axis=axis, keepdims=keepdims
        )
    except tensorflow.errors.InvalidArgumentError as e:
        raise Exception(e) from e


def tensorflow__broadcast_inputs(x1, x2):
    x1_, x2_ = x1, x2
    iterables = list, tuple, tuple
    if not isinstance(x1_, iterables):
        x1_, x2_ = x2, x1
    if not isinstance(x1_, iterables):
        return [x1], [x2]
    if not isinstance(x2_, iterables):
        x1 = [x1] * len(x2)
    return x1, x2


def tensorflow_check_equal(x1, x2, inverse=False, message="", as_array=True):
    def eq_fn(x1, x2):
        return x1 == x2 if inverse else x1 != x2

    def comp_fn(x1, x2):
        return tensorflow_any(eq_fn(x1, x2))

    if not as_array:

        def iter_comp_fn(x1_, x2_):
            return any(eq_fn(x1, x2) for x1, x2 in zip(x1_, x2_))

        def comp_fn(x1, x2):
            return iter_comp_fn(*tensorflow__broadcast_inputs(x1, x2))

    eq = comp_fn(x1, x2)
    if inverse and eq:
        raise Exception(f"{x1} must not be equal to {x2}" if message == "" else message)
    elif not inverse and eq:
        raise Exception(f"{x1} must be equal to {x2}" if message == "" else message)


def tensorflow_multiply(
    x1: Union[float, tensorflow.Tensor, tensorflow.Variable],
    x2: Union[float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype if hasattr(x2, "dtype") else tensorflow_default_dtype_bknd()
        )
        if not tensorflow_is_array_bknd(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array_bknd(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    return tensorflow.math.multiply(x1, x2)


def tensorflow_check_gather_nd_input_valid(params, indices, batch_dims):
    if batch_dims >= len(params.shape):
        raise Exception(
            f"batch_dims = {batch_dims} must be less than rank(`params`) = {len(params.shape)}."
        )
    if batch_dims >= len(indices.shape):
        raise Exception(
            f"batch_dims = {batch_dims}  must be less than rank(`indices`) = {len(indices.shape)}."
        )
    if tensorflow_get_item(
        params.shape, slice(0, batch_dims, None)
    ) != tensorflow_get_item(indices.shape, slice(0, batch_dims, None)):
        raise Exception(
            f"batch dimensions must match in `params` and `indices`; saw {tensorflow_get_item(params.shape, slice(0, batch_dims, None))} vs. {tensorflow_get_item(indices.shape, slice(0, batch_dims, None))}"
        )
    if indices.shape[-1] > len(
        tensorflow_get_item(params.shape, slice(batch_dims, None, None))
    ):
        raise Exception(
            f"index innermost dimension length must be <= rank(`params[batch_dims:]`); saw: {indices.shape[-1]} vs. {len(tensorflow_get_item(params.shape, slice(batch_dims, None, None)))} ."
        )


def tensorflow_gather_nd_helper(params, indices):
    indices_shape = tensorflow.shape(indices)
    params_shape = tensorflow.shape(params)
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        tensorflow.math.reduce_prod(params_shape[i + 1 :])
        for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = tensorflow.convert_to_tensor(
        result_dim_sizes_list, dtype=indices.dtype
    )
    implicit_indices_factor = result_dim_sizes[num_index_dims - 1]
    flat_params = tensorflow.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = tensorflow.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = tensorflow.reshape(
        tensorflow.reduce_sum(indices * indices_scales, -1, keepdims=True), (-1, 1)
    )
    indices_for_flat_tiled = tensorflow.repeat(
        indices_for_flat_tiled, implicit_indices_factor, axis=1
    )
    implicit_indices = tensorflow.repeat(
        tensorflow.expand_dims(tensorflow.range(implicit_indices_factor), 0),
        indices_for_flat_tiled.shape[0],
        axis=0,
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = tensorflow.reshape(indices_for_flat, (-1,))
    flat_gather = tensorflow.gather(flat_params, flat_indices_for_flat)
    res = tensorflow.reshape(
        flat_gather,
        tensorflow.concat([indices_shape[:-1], params_shape[num_index_dims:]], 0),
    )
    return res


@tensorflow_handle_array_like_without_promotion
def tensorflow_gather_nd(
    params: Union[tensorflow.Tensor, tensorflow.Variable],
    indices: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    batch_dims: int = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    tensorflow_check_gather_nd_input_valid(params, indices, batch_dims)
    try:
        return tensorflow.gather_nd(params, indices, batch_dims=batch_dims)
    except Exception:
        batch_dims %= len(params.shape)
        result = []
        if batch_dims == 0:
            result = tensorflow_gather_nd_helper(params, indices)
        else:
            for b in range(batch_dims):
                if b == 0:
                    zip_list = list(zip(params, indices))
                else:
                    zip_list = [
                        (p, i)
                        for z in [zip(p1, i1) for p1, i1 in zip_list]
                        for p, i in z
                    ]
            for z in zip_list:
                p, i = z[0], z[1]
                r = tensorflow_gather_nd_helper(p, i)
                result.append(r)
            result = tensorflow.stack(result)
            result = tensorflow.reshape(
                result,
                tensorflow.concat([params.shape[0:batch_dims], result.shape[1:]], 0),
            )
        return result


def tensorflow__is_variable_bknd(x, exclusive=False, to_ignore=None):
    x = x
    return tensorflow_nested_map_bknd(
        lambda x: tensorflow_is_variable(x, exclusive=exclusive),
        x,
        include_derived=True,
        shallow=False,
        to_ignore=to_ignore,
    )


def tensorflow_inplace_update(
    x: Union[tensorflow.Tensor, tensorflow.Tensor],
    val: Union[tensorflow.Tensor, tensorflow.Tensor],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
):
    if tensorflow_is_array_bknd(x) and tensorflow_is_array_bknd(val):
        if keep_input_dtype:
            val = tensorflow_astype(val, x.dtype)
        (x_native, val_native), _ = (x, val), "_"
        if tensorflow__is_variable_bknd(x_native):
            x_native.assign(val_native)
            if tensorflow_is_ivy_array_bknd(x):
                x = x_native
            else:
                x = tensorflow.convert_to_tensor(x_native)
        else:
            x = x_native
        return x
    else:
        return val


def tensorflow_scatter_nd(
    indices: Union[tensorflow.Tensor, tensorflow.Variable],
    updates: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    shape: Optional[Union[tf.TensorShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    updates_dtype = updates.dtype
    if tensorflow_exists_bknd(out):
        dtype = tensorflow_promote_types_bknd(out.dtype, updates_dtype)
    updates = tensorflow.cast(
        updates,
        (
            tensorflow_as_native_dtype(dtype)
            if tensorflow_exists_bknd(out)
            else updates_dtype
        ),
    )
    expected_shape = (
        list(tensorflow.shape(indices)[:-1])
        + list(out.shape[tensorflow.shape(indices)[-1] :])
        if tensorflow_exists_bknd(out)
        else list(tensorflow.shape(indices)[:-1])
        + list(shape[tensorflow.shape(indices)[-1] :])
    )
    updates = tensorflow__broadcast_to_bknd(updates, expected_shape)
    if len(updates.shape) == 0:
        indices = tensorflow.expand_dims(indices, 0)
        updates = tensorflow.expand_dims(updates, 0)
    target = out
    target_given = tensorflow_exists_bknd(target)
    if tensorflow_exists_bknd(shape) and target_given:
        tensorflow_check_equal(tuple(target.shape), tuple(shape), as_array=False)
    if not target_given:
        shape = list(shape) if tensorflow_exists_bknd(shape) else list(out.shape)
        target = tensorflow.zeros(shape, dtype=updates.dtype)
    if reduction == "sum":
        res = tensorflow.tensor_scatter_nd_add(target, indices, updates)
    elif reduction == "min":
        res = tensorflow.tensor_scatter_nd_min(target, indices, updates)
    elif reduction == "max":
        res = tensorflow.tensor_scatter_nd_max(target, indices, updates)
    elif reduction == "mul":
        updates = tensorflow_multiply(tensorflow_gather_nd(target, indices), updates)
        res = tensorflow.tensor_scatter_nd_update(target, indices, updates)
    elif reduction == "replace":
        res = tensorflow.tensor_scatter_nd_update(target, indices, updates)
    else:
        raise Exception(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max", "mul" or "replace"'
        )
    if tensorflow_exists_bknd(out):
        return tensorflow_inplace_update(out, res)
    return res


def tensorflow_handle_set_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, val, **kwargs):
        try:
            inp.__setitem__(query, val)
            res = inp
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, val, **kwargs)
        return res

    return wrapper


@tensorflow_handle_set_item
def tensorflow_set_item_bknd(
    x: Union[tensorflow.Tensor, tf.Tensor],
    query: Union[tensorflow.Tensor, tf.Tensor, Tuple],
    val: Union[tensorflow.Tensor, tf.Tensor],
    /,
    *,
    copy: Optional[bool] = False,
):
    if isinstance(query, (list, tuple)) and any(
        [(q is Ellipsis or isinstance(q, slice) and q.stop is None) for q in query]
    ):
        x_stop_gradient = tensorflow_stop_gradient(x, preserve_type=False)
        np_array = x_stop_gradient.numpy()
        val_stop_gradient = tensorflow_stop_gradient(val, preserve_type=False)
        np_array = tensorflow_set_item_bknd(
            np_array, query, np.asarray(val_stop_gradient)
        )
        return tensorflow_asarray(np_array)
    if copy:
        x = tensorflow_copy_array(x)
    if not tensorflow_is_array_bknd(val):
        val = tensorflow_asarray(val)
    if 0 in x.shape or 0 in val.shape:
        return x
    if tensorflow_is_array_bknd(query) and tensorflow_is_bool_dtype_bknd(query):
        if not len(query.shape):
            query = tensorflow_tile(query, (x.shape[0],))
        indices = tensorflow_nonzero(query, as_tuple=False)
    else:
        indices, target_shape, _ = tensorflow__parse_query_bknd(
            query, tensorflow_shape(x, as_array=True), scatter=True
        )
        if indices is None:
            return x
    val = tensorflow_astype_bknd_(val, x.dtype)
    ret = tensorflow_scatter_nd(indices, val, reduction="replace", out=x)
    return ret


@tensorflow_handle_array_like_without_promotion
def tensorflow_real(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.math.real(x)


def tensorflow_real_bknd_(self):
    return tensorflow_real(self)


@tensorflow_handle_array_like_without_promotion
def tensorflow_imag(
    val: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.math.imag(val, name=None)


def tensorflow_imag_bknd_(self):
    return tensorflow_imag(self)


def tensorflow__check_complex128_bknd(input):
    if tensorflow_is_array_bknd(input):
        return tensorflow_dtype(input) == "complex128"
    elif isinstance(input, np.ndarray):
        return str(input.dtype) == "complex128"
    if hasattr(input, "real") and hasattr(input, "imag"):
        return tensorflow__check_float64_bknd(
            tensorflow_real_bknd_(input)
        ) and tensorflow__check_float64_bknd(tensorflow_imag_bknd_(input))
    return False


def tensorflow_default_complex_dtype_bknd(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    complex_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_complex_dtype_stack
    if tensorflow_exists_bknd(complex_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(complex_dtype)
        return str(tensorflow_as_ivy_dtype(complex_dtype))
    as_native = tensorflow_default_bknd(as_native, False)
    if tensorflow_exists_bknd(input):
        if tensorflow_is_array_bknd(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere_bknd(
                input,
                lambda x: tensorflow__check_complex128_bknd(x),
                stop_after_n_found=1,
            ):
                ret = tf.complex128
            elif not default_complex_dtype_stack:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_complex_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "complex64"
            else:
                ret = default_complex_dtype_stack[-1]
        elif isinstance(input, Number):
            if tensorflow__check_complex128_bknd(input):
                ret = tf.complex128
            elif not default_complex_dtype_stack:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_complex_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "complex64"
            else:
                ret = default_complex_dtype_stack[-1]
    elif not default_complex_dtype_stack:
        def_dtype = tensorflow_default_dtype_bknd()
        if tensorflow_is_complex_dtype_bknd(def_dtype):
            ret = def_dtype
        else:
            ret = "complex64"
    else:
        ret = default_complex_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype(ret))


def tensorflow_default_dtype_bknd(
    *,
    dtype: Optional[Union[str, str]] = None,
    item: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    as_native: bool = False,
):
    if tensorflow_exists_bknd(dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(dtype)
        return tensorflow_as_ivy_dtype(dtype)
    as_native = tensorflow_default_bknd(as_native, False)
    if tensorflow_exists_bknd(item):
        if hasattr(item, "override_dtype_check"):
            return item.override_dtype_check()
        elif isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif tensorflow_is_complex_dtype_bknd(item):
            return tensorflow_default_complex_dtype_bknd(
                input=item, as_native=as_native
            )
        elif tensorflow_is_float_dtype_bknd(item):
            return tensorflow_default_float_dtype_bknd(input=item, as_native=as_native)
        elif tensorflow_is_uint_dtype_bknd(item):
            return tensorflow_default_int_dtype_bknd(input=item, as_native=as_native)
        elif tensorflow_is_int_dtype_bknd(item):
            return tensorflow_default_int_dtype_bknd(input=item, as_native=as_native)
        elif as_native:
            return tensorflow_as_native_dtype("bool")
        else:
            return "bool"
    global default_dtype_stack
    if not default_dtype_stack:
        global default_float_dtype_stack
        if default_float_dtype_stack:
            ret = default_float_dtype_stack[-1]
        else:
            ret = "float32"
    else:
        ret = default_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return tensorflow_as_ivy_dtype(ret)


def tensorflow_default_float_dtype_bknd(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    float_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_float_dtype_stack
    if tensorflow_exists_bknd(float_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(float_dtype)
        return str(tensorflow_as_ivy_dtype(float_dtype))
    as_native = tensorflow_default_bknd(as_native, False)
    if tensorflow_exists_bknd(input):
        if tensorflow_is_array_bknd(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere_bknd(
                input, lambda x: tensorflow__check_float64_bknd(x), stop_after_n_found=1
            ):
                ret = tf.float64
            elif not default_float_dtype_stack:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_float_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "float32"
            else:
                ret = default_float_dtype_stack[-1]
        elif isinstance(input, Number):
            if tensorflow__check_float64_bknd(input):
                ret = tf.float64
            elif not default_float_dtype_stack:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_float_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "float32"
            else:
                ret = default_float_dtype_stack[-1]
    elif not default_float_dtype_stack:
        def_dtype = tensorflow_default_dtype_bknd()
        if tensorflow_is_float_dtype_bknd(def_dtype):
            ret = def_dtype
        else:
            ret = "float32"
    else:
        ret = default_float_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype(ret))


def tensorflow_as_ivy_dtype(
    dtype_in: Union[tensorflow.DType, str, int, float, complex, bool, np.dtype], /
):
    if dtype_in is int:
        return tensorflow_default_int_dtype_bknd()
    if dtype_in is float:
        return tensorflow_default_float_dtype_bknd()
    if dtype_in is complex:
        return tensorflow_default_complex_dtype_bknd()
    if dtype_in is bool:
        return str("bool")
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if isinstance(dtype_in, str):
        if dtype_in in native_dtype_dict:
            dtype_str = dtype_in
        else:
            raise Exception(
                f"Cannot convert to ivy dtype. {dtype_in} is not supported by TensorFlow backend."
            )
    else:
        dtype_str = ivy_dtype_dict[dtype_in]
    if "uint" in dtype_str:
        return str(dtype_str)
    elif "int" in dtype_str:
        return str(dtype_str)
    elif "float" in dtype_str:
        return str(dtype_str)
    elif "complex" in dtype_str:
        return str(dtype_str)
    elif "bool" in dtype_str:
        return str("bool")
    else:
        raise Exception(f"Cannot recognize {dtype_str} as a valid Dtype.")


def tensorflow_default_int_dtype_bknd(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    int_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_int_dtype_stack
    if tensorflow_exists_bknd(int_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(int_dtype)
        return str(tensorflow_as_ivy_dtype(int_dtype))
    as_native = tensorflow_default_bknd(as_native, False)
    if tensorflow_exists_bknd(input):
        if tensorflow_is_array_bknd(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, tuple):
            ret = tensorflow_default_int_dtype_bknd()
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere_bknd(
                input,
                lambda x: (
                    tensorflow_dtype(x) == "uint64"
                    if tensorflow_is_array_bknd(x)
                    else x > 9223372036854775807 and x != math.inf
                ),
                stop_after_n_found=1,
            ):
                ret = tf.uint64
            elif tensorflow_nested_argwhere_bknd(
                input,
                lambda x: (
                    tensorflow_dtype(x) == "int64"
                    if tensorflow_is_array_bknd(x)
                    else x > 2147483647 and x != math.inf
                ),
                stop_after_n_found=1,
            ):
                ret = tf.int64
            elif not default_int_dtype_stack:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_int_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "int32"
            else:
                ret = default_int_dtype_stack[-1]
        elif isinstance(input, Number):
            if input > 9223372036854775807 and input != math.inf and backend != "torch":
                ret = tf.uint64
            elif input > 2147483647 and input != math.inf:
                ret = tf.int64
            elif not default_int_dtype_stack:
                def_dtype = tensorflow_default_dtype_bknd()
                if tensorflow_is_int_dtype_bknd(def_dtype):
                    ret = def_dtype
                else:
                    ret = "int32"
            else:
                ret = default_int_dtype_stack[-1]
    elif not default_int_dtype_stack:
        def_dtype = tensorflow_default_dtype_bknd()
        if tensorflow_is_int_dtype_bknd(def_dtype):
            ret = def_dtype
        else:
            ret = "int32"
    else:
        ret = default_int_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype(ret))


def tensorflow_as_native_dtype(
    dtype_in: Union[tensorflow.DType, str, bool, int, float, np.dtype],
):
    if dtype_in is int:
        return tensorflow_default_int_dtype_bknd(as_native=True)
    if dtype_in is float:
        return tensorflow_default_float_dtype_bknd(as_native=True)
    if dtype_in is complex:
        return tensorflow_default_complex_dtype_bknd(as_native=True)
    if dtype_in is bool:
        return tensorflow.bool
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict:
        return native_dtype_dict[str(dtype_in)]
    else:
        raise Exception(
            f"Cannot convert to TensorFlow dtype. {dtype_in} is not supported by TensorFlow."
        )


def tensorflow_dtype(
    x: Union[tensorflow.Tensor, tensorflow.Variable, np.ndarray],
    *,
    as_native: bool = False,
):
    if as_native:
        return tensorflow_as_native_dtype(x.dtype)
    return tensorflow_as_ivy_dtype(x.dtype)


def tensorflow_is_bool_dtype_bknd(
    dtype_in: Union[str, str, tensorflow.Tensor, tf.Tensor, Number], /
):
    if tensorflow_is_array_bknd(dtype_in):
        dtype_in = tensorflow_dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "bool" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (bool, np.bool_)) and not isinstance(dtype_in, bool)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return bool(
            tensorflow_nested_argwhere_bknd(
                dtype_in, lambda x: isinstance(x, (bool, np.bool_)) and x is not int
            )
        )
    return "bool" in tensorflow_as_ivy_dtype(dtype_in)


def tensorflow_handle_get_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, **kwargs):
        try:
            res = inp.__getitem__(query)
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, **kwargs)
        return res

    return wrapper


@tensorflow_handle_get_item
def tensorflow_get_item(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    query: Union[tensorflow.Tensor, tensorflow.Variable, Tuple],
    *,
    copy: Optional[bool] = None,
):
    if (
        tensorflow_is_array_bknd(query)
        and tensorflow_is_bool_dtype_bknd(query)
        and not len(query.shape)
    ):
        return tensorflow.expand_dims(x, 0)
    return x[query]


@tensorflow_handle_methods
def tensorflow_split_frnt(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        split_size_or_sections = [split_size] * (
            tensorflow_get_item(tensor.shape, dim) // split_size
        )
        if tensorflow_get_item(tensor.shape, dim) % split_size:
            split_size_or_sections.append(
                tensorflow_get_item(tensor.shape, dim) % split_size
            )
    return tuple(
        tensorflow_split(
            tensor,
            num_or_size_splits=split_size_or_sections,
            axis=dim,
            with_remainder=True,
        )
    )


@tensorflow_handle_methods
def tensorflow_split_frnt_(tensor, split_size, dim=0):
    return tensorflow_split_frnt(tensor, split_size, dim)


@tensorflow_handle_methods_1
def tensorflow_add(
    x1: Union[float, tensorflow.Tensor, tensorflow.Variable],
    x2: Union[float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype if hasattr(x2, "dtype") else tensorflow_default_dtype_bknd()
        )
        if not tensorflow_is_array_bknd(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array_bknd(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    if x1.dtype.is_bool and x2.dtype.is_bool:
        return tensorflow.math.logical_or(x1, x2)
    if alpha not in (1, None):
        x2 = tensorflow_multiply(x2, alpha)
    return tensorflow.add(x1, x2)


@tensorflow_handle_methods
def tensorflow_add_frnt(input, other, *, alpha=1, out=None):
    return tensorflow_add(input, other, alpha=alpha, out=out)


@tensorflow_handle_methods
def tensorflow_add_frnt_(tensor, other, *, alpha=1):
    return tensorflow_add_frnt(tensor, other, alpha=alpha)


def tensorflow_ndim_bknd_(self):
    return len(tuple(self.shape))


@tensorflow_handle_array_like_without_promotion
def tensorflow_permute_dims(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.transpose(x, perm=axes)


def tensorflow__handle_padding_bknd(x, strides, filters, padding):
    if isinstance(padding, str) and padding.upper() == "SAME":
        if x % strides == 0:
            pad = max(filters - strides, 0)
        else:
            pad = max(filters - x % strides, 0)
    else:
        pad = 0
    return pad


def tensorflow__output_ceil_shape_bknd(w, f, p, s):
    return math.ceil((w - f + p) / s) + 1


def tensorflow__padding_ceil_mode_bknd(
    w: int, f: int, p: Tuple[int], s: int, return_added_padding: Optional[bool] = False
):
    remaining_pixels = (w - f + sum(p)) % s
    added_padding = 0
    if remaining_pixels <= p[1] and s + p[1] - remaining_pixels >= f:
        return (p, added_padding) if return_added_padding else p
    if s > 1 and remaining_pixels != 0 and f > 1:
        input_size = w + sum(p)
        if input_size - remaining_pixels - (f - 1) + s > input_size:
            return (p, added_padding) if return_added_padding else p
        output_shape = tensorflow__output_ceil_shape_bknd(w, f, sum(p), s)
        new_pad = (output_shape - 1) * s + f - w
        added_padding = new_pad - sum(p)
        p = p[0], p[1] + added_padding
    if return_added_padding:
        return p, added_padding
    return p


def tensorflow__handle_manual_pad_avg_pool(
    x, kernel, strides, padding, ceil_mode, dims
):
    if isinstance(padding, str):
        pad_specific = [
            tensorflow__handle_padding_bknd(
                x.shape[i + 1], strides[i], kernel[i], padding
            )
            for i in range(dims)
        ]
        padding = [
            (pad_specific[i] // 2, pad_specific[i] - pad_specific[i] // 2)
            for i in range(dims)
        ]
    else:
        if isinstance(padding, int):
            padding = [(padding,) * 2] * dims
        pad_specific = [sum(padding[i]) for i in range(dims)]
    c = []
    if ceil_mode:
        for i in range(dims):
            padding[i], c_i = tensorflow__padding_ceil_mode_bknd(
                x.shape[i + 1], kernel[i], padding[i], strides[i], True
            )
            c.append(c_i)
            pad_specific[i] = sum(padding[i])
    return padding, pad_specific, c


def tensorflow_map_bknd(
    fn: Callable,
    constant: Optional[Dict[str, Any]] = None,
    unique: Optional[Dict[str, Iterable[Any]]] = None,
    mean: bool = False,
):
    c = tensorflow_default_bknd(constant, {})
    u = tensorflow_default_bknd(unique, {})
    rets = [
        r
        for r in _map(
            lambda *uv: fn(**dict(**c, **dict(zip(u.keys(), uv)))), *u.values()
        )
    ]
    if mean:
        rets = sum(rets) / len(rets)
    return rets


def tensorflow__get_num_padded_values_bknd(i, p, n, k, s):
    current_index = s * i
    left_padding = p // 2
    return max(0, left_padding - current_index) + max(
        0, current_index + k - n - left_padding
    )


def tensorflow_avg_pool2d(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if isinstance(kernel, int):
        kernel = (kernel,) * 2
    elif len(kernel) == 1:
        kernel = (kernel[0],) * 2
    if isinstance(strides, int):
        strides = (strides,) * 2
    elif len(strides) == 1:
        strides = (strides[0],) * 2
    if data_format == "NCHW":
        x = tensorflow.transpose(x, (0, 2, 3, 1))
    manual_padding = False
    if not isinstance(padding, str) or ceil_mode or count_include_pad:
        padding, pad_specific, c = tensorflow__handle_manual_pad_avg_pool(
            x, kernel, strides, padding, ceil_mode, 2
        )
        x = tensorflow.pad(x, [(0, 0), *padding, (0, 0)], constant_values=0)
        manual_padding = True
        padding = "VALID"
    if divisor_override is not None:
        res = tensorflow.nn.depthwise_conv2d(
            x,
            tensorflow.ones(kernel + (x.shape[-1], 1)),
            (1,) + strides + (1,),
            padding,
        )
        res = res / divisor_override
    else:
        res = tensorflow.nn.avg_pool2d(x, kernel, strides, padding)
    if manual_padding and not count_include_pad or ceil_mode and not divisor_override:
        if not count_include_pad:
            num_padded_values = [
                tensorflow.convert_to_tensor(
                    tensorflow_map_bknd(
                        tensorflow__get_num_padded_values_bknd,
                        constant={
                            "p": pad_specific[i],
                            "n": x.shape[i + 1] - pad_specific[i],
                            "k": kernel[i],
                            "s": strides[i],
                        },
                        unique={"i": tensorflow.range(res.shape[i + 1])},
                    ),
                    dtype=res.dtype,
                )
                for i in range(2)
            ]
        else:
            num_padded_values = []
            for i in range(2):
                num_pad = tensorflow.scatter_nd(
                    tensorflow.constant([[res.shape[i + 1] - 1]]),
                    tensorflow.constant([c[i]], dtype=res.dtype),
                    tensorflow.constant([res.shape[i + 1]], dtype=tensorflow.int32),
                )
                num_padded_values.append(num_pad)
        num_padded_values1 = num_padded_values[0][:, None]
        num_padded_values2 = num_padded_values[1][None, :]
        num_padded_values = (
            num_padded_values1 * kernel[1]
            + num_padded_values2 * kernel[0]
            - num_padded_values1 * num_padded_values2
        )
        kernel_mul = tensorflow.cast(tensorflow.math.reduce_prod(kernel), res.dtype)
        res = (
            kernel_mul
            * res
            / (kernel_mul - tensorflow.expand_dims(num_padded_values, -1))
        )
    if data_format == "NCHW":
        return tensorflow.transpose(res, (0, 3, 1, 2))
    return res


@tensorflow_handle_methods_1
@tensorflow_handle_array_like_without_promotion
def tensorflow_sort(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    direction = "DESCENDING" if descending else "ASCENDING"
    x = tensorflow.convert_to_tensor(x)
    ret = tensorflow.sort(x, axis=axis, direction=direction)
    return ret


@tensorflow_handle_methods_1
def tensorflow_sort_bknd_(
    self: tensorflow.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[tensorflow.Tensor] = None,
):
    return tensorflow_sort(
        self, axis=axis, descending=descending, stable=stable, out=out
    )


@tensorflow_handle_array_like_without_promotion
def tensorflow_squeeze(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    copy: Optional[bool] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if isinstance(axis, int):
        if tensorflow_any(x.shape[axis] > 1):
            raise ValueError(f"{x.shape[axis]} must be lesser than or equal to 1")
        ret = tensorflow.squeeze(x, axis)
    elif axis is None:
        ret = tensorflow.squeeze(x)
    else:
        if isinstance(axis, tuple):
            axis = list(axis)
        normalise_axis = [
            (len(x.shape) - abs(element) if element < 0 else element)
            for element in axis
        ]
        tensorflow_sort_bknd_(normalise_axis)
        axis_updated_after_squeeze = [
            (dim - key) for key, dim in enumerate(normalise_axis)
        ]
        for i in axis_updated_after_squeeze:
            if x.shape[i] > 1:
                raise ValueError(
                    f"Expected dimension of size 1, but found dimension size {x.shape[i]}"
                )
            else:
                x = tensorflow.squeeze(x, i)
        ret = x
    return ret


@tensorflow_handle_array_like_without_promotion
def tensorflow_trunc(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    ret = x
    if not tensorflow_is_array_bknd(x):
        raise Exception("Input must be array")
    elif "int" not in str(x.dtype):
        if ret.get_shape().ndims != 0:
            ret = tensorflow.tensor_scatter_nd_update(
                x,
                tensorflow.where(tensorflow.greater_equal(x, 0)),
                tensorflow.math.floor(x[x >= 0]),
            )
            ret = tensorflow.tensor_scatter_nd_update(
                ret,
                tensorflow.where(tensorflow.less(x, 0)),
                tensorflow.math.ceil(x[x < 0]),
            )
        else:
            ret = (tensorflow.math.floor if ret >= 0 else tensorflow.math.ceil)(ret)
    return ret


def tensorflow_divide(
    x1: Union[float, tensorflow.Tensor, tensorflow.Variable],
    x2: Union[float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype if hasattr(x2, "dtype") else tensorflow_default_dtype_bknd()
        )
        if not tensorflow_is_array_bknd(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array_bknd(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    ret = tensorflow.experimental.numpy.divide(x1, x2)
    if tensorflow_is_float_dtype_bknd(x1.dtype) or tensorflow_is_complex_dtype_bknd(
        x1.dtype
    ):
        ret = tensorflow.cast(ret, dtype=x1.dtype)
    else:
        ret = tensorflow.cast(
            ret, dtype=tensorflow_default_float_dtype_bknd(as_native=True)
        )
    return ret


@tensorflow_handle_array_like_without_promotion
def tensorflow_trunc_divide_bknd(
    x1: Union[float, tensorflow.Tensor, tf.Tensor],
    x2: Union[float, tensorflow.Tensor, tf.Tensor],
    /,
    *,
    out: Optional[tensorflow.Tensor] = None,
):
    return tensorflow_trunc(tensorflow_divide(x1, x2), out=out)


@tensorflow_infer_dtype
@tensorflow_handle_array_like_without_promotion
def tensorflow_full_like(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    fill_value: Number,
    *,
    dtype: tensorflow.DType,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.experimental.numpy.full_like(x, fill_value, dtype=dtype)


def tensorflow_minimum(
    x1: Union[tensorflow.Tensor, tensorflow.Variable],
    x2: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    use_where: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype if hasattr(x2, "dtype") else tensorflow_default_dtype_bknd()
        )
        if not tensorflow_is_array_bknd(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array_bknd(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    return tensorflow.math.minimum(x1, x2)


def tensorflow__compute_idx_bknd(in_size, out_size, device):
    out_range = tensorflow_arange(out_size, device=device, dtype=tf.int64)
    i0 = tensorflow_astype_bknd_(
        tensorflow_trunc_divide_bknd(out_range * in_size, out_size), tf.int64
    )
    maxlength = in_size // out_size + 1
    in_size_mod = in_size % out_size
    adaptive = in_size_mod != 0 and out_size % in_size_mod != 0
    if adaptive:
        maxlength = maxlength + 1
    elif in_size_mod == 0:
        maxlength = maxlength - 1
    range_max = tensorflow_arange(maxlength, device=device, dtype=tf.int64)
    idx = tensorflow_expand_dims(i0, axis=-1) + range_max
    if adaptive:
        maxval = tensorflow_full_like(idx, fill_value=in_size - 1)
        idx = tensorflow_minimum(idx, maxval)
        i1 = tensorflow_astype_bknd_(
            tensorflow_trunc_divide_bknd(
                (out_range + 1) * in_size + out_size - 1, out_size
            ),
            tf.int64,
        )
        length = i1 - i0
    else:
        length = maxlength
    return idx, length, range_max, adaptive


def tensorflow__expand_to_dim_bknd(x, dim):
    for _ in range(dim - len(x.shape)):
        x = tensorflow_expand_dims(x, axis=-1)
    return x


@tensorflow_infer_dtype
@tensorflow_handle_array_like_without_promotion
def tensorflow_mean(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    *,
    dtype: Optional[tensorflow.DType] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is not None:
        x = tensorflow_astype(x, dtype)
    return tensorflow.math.reduce_mean(x, axis=axis, keepdims=keepdims)


def tensorflow_greater_equal(
    x1: Union[float, tensorflow.Tensor, tensorflow.Variable],
    x2: Union[float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype if hasattr(x2, "dtype") else tensorflow_default_dtype_bknd()
        )
        if not tensorflow_is_array_bknd(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array_bknd(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    return tensorflow.math.greater_equal(x1, x2)


def tensorflow__mask_bknd(vals, length, range_max, dim, mask_value=0.0):
    if isinstance(length, int):
        return vals, length
    else:
        assert dim < 0
        mask = tensorflow_greater_equal(
            range_max, tensorflow_expand_dims(length, axis=-1)
        )
        if dim == -2:
            mask = tensorflow__expand_to_dim_bknd(mask, 4)
        vals = tensorflow_where(
            mask, tensorflow_asarray(mask_value, device=vals.device), vals
        )
        length = tensorflow__expand_to_dim_bknd(length, -dim)
        return vals, length


@tensorflow_handle_array_like_without_promotion
def tensorflow_adaptive_avg_pool2d_bknd(
    input: Union[tensorflow.Tensor, tf.Tensor],
    output_size: Union[Sequence[int], int],
    /,
    *,
    data_format: str = "NHWC",
):
    squeeze = False
    if tensorflow_ndim_bknd_(input) == 3:
        input = tensorflow_expand_dims(input, axis=0)
        squeeze = True
    elif tensorflow_ndim_bknd_(input) != 4:
        raise Exception(
            f"Got {len(input.shape)}D input, but only 3D and 4D inputs are supported."
        )
    permuted_input = False
    if data_format == "NHWC":
        input = tensorflow_permute_dims(
            input,
            (
                0,
                tensorflow_ndim_bknd_(input) - 1,
                *range(1, tensorflow_ndim_bknd_(input) - 1),
            ),
        )
        data_format = "NCHW"
        permuted_input = True
    if isinstance(output_size, int):
        output_size = output_size, output_size
    if all(i_s % o_s == 0 for i_s, o_s in zip(input.shape[-2:], output_size)):
        stride = tuple(i_s // o_s for i_s, o_s in zip(input.shape[-2:], output_size))
        kernel_size = stride
        pooled_output = tensorflow_avg_pool2d(
            input, kernel_size, stride, "VALID", data_format="NCHW"
        )
        pooled_output = (
            tensorflow_permute_dims(
                pooled_output, (0, *range(2, tensorflow_ndim_bknd_(input)), 1)
            )
            if permuted_input
            else pooled_output
        )
        if squeeze:
            return tensorflow_squeeze(pooled_output, axis=0)
        return pooled_output
    idxh, length_h, range_max_h, adaptive_h = tensorflow__compute_idx_bknd(
        input.shape[-2], output_size[-2], input.device
    )
    idxw, length_w, range_max_w, adaptive_w = tensorflow__compute_idx_bknd(
        input.shape[-1], output_size[-1], input.device
    )
    vals = tensorflow_get_item(
        input, (..., tensorflow__expand_to_dim_bknd(idxh, 4), idxw)
    )
    if not adaptive_h and not adaptive_w:
        ret = tensorflow_mean(vals, axis=(-3, -1))
        ret = (
            tensorflow_permute_dims(
                ret, (0, *range(2, tensorflow_ndim_bknd_(input)), 1)
            )
            if permuted_input
            else ret
        )
        ret = tensorflow_squeeze(ret, axis=0) if squeeze else ret
        return ret
    vals, length_h = tensorflow__mask_bknd(vals, length_h, range_max_h, dim=-2)
    vals, length_w = tensorflow__mask_bknd(vals, length_w, range_max_w, dim=-1)
    ret = None
    for i, j in itertools.product(range(vals.shape[-3]), range(vals.shape[-1])):
        if ret is None:
            ret = tensorflow_get_item(vals, (..., i, slice(None, None, None), j))
        else:
            ret = ret + tensorflow_get_item(vals, (..., i, slice(None, None, None), j))
    pooled_output = ret / tensorflow_astype_bknd_(length_h * length_w, vals.dtype)
    pooled_output = (
        tensorflow_permute_dims(
            pooled_output, (0, *range(2, tensorflow_ndim_bknd_(input)), 1)
        )
        if permuted_input
        else pooled_output
    )
    pooled_output = (
        tensorflow_squeeze(pooled_output, axis=0) if squeeze else pooled_output
    )
    return pooled_output


def tensorflow_adaptive_avg_pool2d_frnt(input, output_size):
    return tensorflow_adaptive_avg_pool2d_bknd(input, output_size, data_format="NHWC")


def tensorflow_retrieve_object(frame, name):
    if name is None:
        return name
    names = tensorflow_split_bknd_(name, ".")
    obj = frame.f_locals.get(names[0]) or frame.f_globals.get(names[0])
    if obj is None:
        return None
    for attr in names[1:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
    return obj


def tensorflow_get_next_func(obj):
    from .tensorflow_CallVisitor import tensorflow_CallVisitor

    stack = inspect.stack()
    for frame_info in stack:
        if frame_info == obj._previous_frame_info:
            calling_frame = frame_info.frame
            break
    else:
        return None
    if "Sequential" in frame_info.filename:
        try:
            self_seq = calling_frame.f_locals["self"]
            idx = calling_frame.f_locals["i"]
            next_func = tensorflow_get_item(self_seq, idx + 1)
            return next_func
        except IndexError:
            for frame_info in tensorflow_get_item(
                stack, slice(stack.index(frame_info) + 1, None, None)
            ):
                if frame_info == self_seq._previous_frame_info:
                    calling_frame = frame_info.frame
                    break
            else:
                return None
    lines, start_line_no = inspect.getsourcelines(calling_frame)
    current_line_no = calling_frame.f_lineno
    relative_line_no = current_line_no - start_line_no
    try:
        next_line = tensorflow_get_item(lines, relative_line_no + 1).strip()
        tree = ast.parse(next_line)
        visitor = tensorflow_CallVisitor()
        visitor.visit(tree)
        next_call_str = visitor.func_name
    except Exception:
        next_call_str = ""
    next_func = tensorflow_retrieve_object(calling_frame, next_call_str)
    return next_func


def tensorflow_apply_transpose(input, transpose, pt_to_tf=True):
    from .tensorflow_TransposeType import tensorflow_TransposeType

    if transpose is tensorflow_TransposeType.NO_TRANSPOSE:
        return input
    if transpose is tensorflow_TransposeType.CONV1D:
        axes = (0, 2, 1) if pt_to_tf else (0, 2, 1)
    elif transpose is tensorflow_TransposeType.CONV2D:
        axes = (0, 2, 3, 1) if pt_to_tf else (0, 3, 1, 2)
    elif transpose is tensorflow_TransposeType.CONV3D:
        axes = (0, 2, 3, 4, 1) if pt_to_tf else (0, 4, 1, 2, 3)
    input = tensorflow_permute_dims(input, axes=axes)
    return input


def tensorflow_handle_transpose_in_input_and_output(fn):
    from .tensorflow_TransposeType import tensorflow_TransposeType

    original_signature = inspect.signature(fn)

    @functools.wraps(fn)
    def transpose_wrapper(self, *args, **kwargs):
        global DATA_FORMAT
        kwargs_call = {
            key: val
            for key, val in kwargs.items()
            if key not in dict(original_signature.parameters)
        }
        fn_args_and_kwargs = {
            key: val for key, val in kwargs.items() if key not in kwargs_call
        }
        fn_args_and_kwargs.update(dict(zip(fn.__code__.co_varnames[1:], args)))
        conv_block_start = lambda f: any(
            substr in f.__qualname__
            for substr in CONV_FUNCS
            + NORM_FUNCS
            + POOL_FUNCS
            + KERAS_CONV_FUNCS
            + KERAS_NORM_FUNCS
            + KERAS_POOL_FUNCS
        )
        next_call_in_seq = tensorflow_get_next_func(self)
        name_of_next_call = (
            next_call_in_seq.__class__.__name__
            if hasattr(next_call_in_seq, "__class__")
            else ""
        )
        conv_block_continued = next_call_in_seq and any(
            substr in name_of_next_call for substr in CONV_BLOCK_FNS
        )
        if DATA_FORMAT == "PT" and conv_block_start(self.__class__):
            input = fn_args_and_kwargs["input"]
            if len(input.shape) > 4:
                transpose = tensorflow_TransposeType.CONV3D
            elif len(input.shape) > 3:
                transpose = tensorflow_TransposeType.CONV2D
            elif len(input.shape) > 2:
                transpose = tensorflow_TransposeType.CONV1D
            else:
                transpose = tensorflow_TransposeType.NO_TRANSPOSE
            fn_args_and_kwargs = tensorflow_set_item_bknd(
                fn_args_and_kwargs,
                "input",
                tensorflow_apply_transpose(input, transpose=transpose, pt_to_tf=True),
            )
            DATA_FORMAT = "TF"
            os.environ = tensorflow_set_item_bknd(
                os.environ, "DATA_FORMAT", "channels_last"
            )
        res = fn(self, **fn_args_and_kwargs)
        if DATA_FORMAT == "TF" and conv_block_continued or DATA_FORMAT == "PT":
            return res
        if len(res.shape) > 4:
            transpose = tensorflow_TransposeType.CONV3D
        elif len(res.shape) > 3:
            transpose = tensorflow_TransposeType.CONV2D
        elif len(res.shape) > 2:
            transpose = tensorflow_TransposeType.CONV1D
        else:
            transpose = tensorflow_TransposeType.NO_TRANSPOSE
        res = tensorflow_apply_transpose(res, transpose=transpose, pt_to_tf=False)
        DATA_FORMAT = "PT"
        os.environ = tensorflow_set_item_bknd(
            os.environ, "DATA_FORMAT", "channels_first"
        )
        return res

    tensorflow_handle_transpose_in_input_and_output.__signature__ = original_signature
    return transpose_wrapper
