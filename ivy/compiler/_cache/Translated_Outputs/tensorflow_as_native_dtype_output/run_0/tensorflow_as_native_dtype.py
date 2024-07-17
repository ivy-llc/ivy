import tensorflow
import numpy as np

from typing import Union

from .tensorflow__helpers import tensorflow_default_complex_dtype
from .tensorflow__helpers import tensorflow_default_float_dtype
from .tensorflow__helpers import tensorflow_default_int_dtype

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


def tensorflow_as_native_dtype(
    dtype_in: Union[tensorflow.DType, str, bool, int, float, np.dtype],
):
    if dtype_in is int:
        return tensorflow_default_int_dtype(as_native=True)
    if dtype_in is float:
        return tensorflow_default_float_dtype(as_native=True)
    if dtype_in is complex:
        return tensorflow_default_complex_dtype(as_native=True)
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
