import tensorflow
import numpy as np

from typing import Union

from .tensorflow__helpers import tensorflow_default_complex_dtype_bknd
from .tensorflow__helpers import tensorflow_default_float_dtype_bknd
from .tensorflow__helpers import tensorflow_default_int_dtype_bknd

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
