# global
from typing import Optional, Union, Sequence, List

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info
from . import backend_version

ivy_dtype_dict = {
    tf.int8: "int8",
    tf.int16: "int16",
    tf.int32: "int32",
    tf.int64: "int64",
    tf.uint8: "uint8",
    tf.uint16: "uint16",
    tf.uint32: "uint32",
    tf.uint64: "uint64",
    tf.bfloat16: "bfloat16",
    tf.float16: "float16",
    tf.float32: "float32",
    tf.float64: "float64",
    tf.complex64: "complex64",
    tf.complex128: "complex128",
    tf.bool: "bool",
}

native_dtype_dict = {
    "int8": tf.int8,
    "int16": tf.int16,
    "int32": tf.int32,
    "int64": tf.int64,
    "uint8": tf.uint8,
    "uint16": tf.uint16,
    "uint32": tf.uint32,
    "uint64": tf.uint64,
    "bfloat16": tf.bfloat16,
    "float16": tf.float16,
    "float32": tf.float32,
    "float64": tf.float64,
    "complex64": tf.complex64,
    "complex128": tf.complex128,
    "bool": tf.bool,
}


class Finfo:
    def __init__(self, tf_finfo: tf.experimental.numpy.finfo):
        self._tf_finfo = tf_finfo

    def __repr__(self):
        return repr(self._tf_finfo)

    @property
    def bits(self):
        return self._tf_finfo.bits

    @property
    def eps(self):
        return float(self._tf_finfo.eps)

    @property
    def max(self):
        return float(self._tf_finfo.max)

    @property
    def min(self):
        return float(self._tf_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._tf_finfo.tiny)


class Bfloat16Finfo:
    def __init__(self):
        self.resolution = 0.01
        self.bits = 16
        self.eps = 0.0078125
        self.max = 3.38953e38
        self.min = -3.38953e38
        self.tiny = 1.17549e-38

    def __repr__(self):
        return "finfo(resolution={}, min={}, max={}, dtype={})".format(
            self.resolution, self.min, self.max, "bfloat16"
        )


# Array API Standard #
# -------------------#


def astype(
    x: Union[tf.Tensor, tf.Variable],
    dtype: tf.DType,
    /,
    *,
    copy: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if x.dtype == dtype:
        return tf.experimental.numpy.copy(x) if copy else x
    return tf.cast(x, dtype)


def broadcast_arrays(
    *arrays: Union[tf.Tensor, tf.Variable],
) -> List[Union[tf.Tensor, tf.Variable]]:
    if len(arrays) > 1:
        desired_shape = tf.broadcast_dynamic_shape(arrays[0].shape, arrays[1].shape)
        if len(arrays) > 2:
            for i in range(2, len(arrays)):
                desired_shape = tf.broadcast_dynamic_shape(
                    desired_shape, arrays[i].shape
                )
    else:
        return [arrays[0]]
    result = []
    for tensor in arrays:
        result.append(tf.broadcast_to(tensor, desired_shape))

    return result


def broadcast_to(
    x: Union[tf.Tensor, tf.Variable],
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if tf.rank(x) > len(shape):
        return tf.broadcast_to(tf.reshape(x, -1), shape)
    return tf.broadcast_to(x, shape)


@_handle_nestable_dtype_info
def finfo(type: Union[DType, str, tf.Tensor, tf.Variable], /) -> Finfo:
    if isinstance(type, tf.Tensor):
        type = type.dtype
    if ivy.as_native_dtype(type) == tf.bfloat16:
        return Finfo(Bfloat16Finfo())
    return Finfo(tf.experimental.numpy.finfo(ivy.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[DType, str, tf.Tensor, tf.Variable], /) -> np.iinfo:
    if isinstance(type, tf.Tensor):
        type = type.dtype
    return tf.experimental.numpy.iinfo(ivy.as_ivy_dtype(type))


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16",)}, backend_version)
def result_type(
    *arrays_and_dtypes: Union[tf.Tensor, tf.Variable, tf.DType],
) -> ivy.Dtype:
    if len(arrays_and_dtypes) <= 1:
        return tf.experimental.numpy.result_type(arrays_and_dtypes)

    result = tf.experimental.numpy.result_type(
        arrays_and_dtypes[0], arrays_and_dtypes[1]
    )
    for i in range(2, len(arrays_and_dtypes)):
        result = tf.experimental.numpy.result_type(result, arrays_and_dtypes[i])
    return as_ivy_dtype(result)


# Extra #
# ------#


def as_ivy_dtype(dtype_in: Union[tf.DType, str, bool, int, float], /) -> ivy.Dtype:
    if dtype_in is int:
        return ivy.default_int_dtype()
    if dtype_in is float:
        return ivy.default_float_dtype()
    if dtype_in is complex:
        return ivy.default_complex_dtype()
    if dtype_in is bool:
        return ivy.Dtype("bool")
    if isinstance(dtype_in, str):
        if dtype_in in native_dtype_dict:
            return ivy.Dtype(dtype_in)
        else:
            raise ivy.exceptions.IvyException(
                "Cannot convert to ivy dtype."
                f" {dtype_in} is not supported by TensorFlow backend."
            )
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


def as_native_dtype(dtype_in: Union[tf.DType, str, bool, int, float], /) -> tf.DType:
    if dtype_in is int:
        return ivy.default_int_dtype(as_native=True)
    if dtype_in is float:
        return ivy.default_float_dtype(as_native=True)
    if dtype_in is complex:
        return ivy.default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return tf.bool
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict.keys():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise ivy.exceptions.IvyException(
            "Cannot convert to TensorFlow dtype."
            f" {dtype_in} is not supported by TensorFlow."
        )


def dtype(x: Union[tf.Tensor, tf.Variable], *, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[tf.DType, str], /) -> int:
    dtype_str = as_ivy_dtype(dtype_in)
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


# ToDo:
# 1. result_type: Add support for bfloat16 with int16
# 2. can_cast : Add support for complex64, complex128
