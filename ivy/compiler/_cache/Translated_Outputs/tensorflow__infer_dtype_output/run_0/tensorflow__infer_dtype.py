import tensorflow

from .tensorflow__helpers import tensorflow_dtype_bits
from .tensorflow__helpers import tensorflow_infer_default_dtype


def tensorflow__infer_dtype(dtype: tensorflow.DType):
    default_dtype = tensorflow_infer_default_dtype(dtype)
    if tensorflow_dtype_bits(dtype) < tensorflow_dtype_bits(default_dtype):
        return default_dtype
    return dtype
