# global
import numpy as np
import tensorflow as tf
from typing import Union, Sequence, List
from tensorflow.python.framework.dtypes import DType

# local
import ivy

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


# Array API Standard #
# -------------------#


def astype(
    x: Union[tf.Tensor, tf.Variable],
    dtype: tf.DType,
    *,
    copy: bool = True,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if copy:
        if x.dtype == dtype:
            new_tensor = tf.experimental.numpy.copy(x)
            return new_tensor
    else:
        if x.dtype == dtype:
            return x
        else:
            new_tensor = tf.experimental.numpy.copy(x)
            new_tensor = tf.cast(new_tensor, dtype)
            return new_tensor
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
    shape: Union[ivy.NativeShape, Sequence[int]],
) -> Union[tf.Tensor, tf.Variable]:
    return tf.broadcast_to(x, shape)


def can_cast(from_: Union[tf.DType, tf.Tensor, tf.Variable], to: tf.DType) -> bool:
    if isinstance(from_, tf.Tensor) or isinstance(from_, tf.Variable):
        from_ = from_.dtype
    from_str = str(from_)
    to_str = str(to)
    if ivy.dtype_bits(to) < ivy.dtype_bits(from_):
        return False
    if ("int" in from_str and "u" not in from_str) and "uint" in to_str:
        return False
    if "bool" in from_str and (("int" in to_str) or ("float" in to_str)):
        return False
    if "int" in from_str and (("float" in to_str) or ("bool" in to_str)):
        return False
    if "float" in from_str and "bool" in to_str:
        return False
    if "float" in from_str and "int" in to_str:
        return False
    if "uint" in from_str and ("int" in to_str and "u" not in to_str):
        if ivy.dtype_bits(to) <= ivy.dtype_bits(from_):
            return False
    return True


def finfo(type: Union[DType, str, tf.Tensor, tf.Variable]) -> Finfo:
    if isinstance(type, tf.Tensor):
        type = type.dtype
    return Finfo(tf.experimental.numpy.finfo(ivy.as_native_dtype(type)))


def iinfo(type: Union[DType, str, tf.Tensor, tf.Variable]) -> np.iinfo:
    if isinstance(type, tf.Tensor):
        type = type.dtype
    return tf.experimental.numpy.iinfo(ivy.as_ivy_dtype(type))


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


def as_ivy_dtype(dtype_in: Union[tf.DType, str]) -> ivy.Dtype:
    if isinstance(dtype_in, str):
        return ivy.Dtype(dtype_in)
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


def as_native_dtype(dtype_in: Union[tf.DType, str]) -> tf.DType:
    if not isinstance(dtype_in, str):
        return dtype_in
    return native_dtype_dict[ivy.Dtype(dtype_in)]


def dtype(x: Union[tf.Tensor, tf.Variable], as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[tf.DType, str]) -> int:
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("tf.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
    )
