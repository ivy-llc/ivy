import tensorflow as tf

from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default_complex_dtype
from .tensorflow__helpers import tensorflow_default_float_dtype
from .tensorflow__helpers import tensorflow_default_int_dtype
from .tensorflow__helpers import tensorflow_default_uint_dtype
from .tensorflow__helpers import tensorflow_is_complex_dtype
from .tensorflow__helpers import tensorflow_is_float_dtype
from .tensorflow__helpers import tensorflow_is_int_dtype
from .tensorflow__helpers import tensorflow_is_uint_dtype


def tensorflow_infer_default_dtype(
    dtype: Union[str, tf.DType, str], as_native: bool = False
):
    if tensorflow_is_complex_dtype(dtype):
        default_dtype = tensorflow_default_complex_dtype(as_native=as_native)
    elif tensorflow_is_float_dtype(dtype):
        default_dtype = tensorflow_default_float_dtype(as_native=as_native)
    elif tensorflow_is_uint_dtype(dtype):
        default_dtype = tensorflow_default_uint_dtype(as_native=as_native)
    elif tensorflow_is_int_dtype(dtype):
        default_dtype = tensorflow_default_int_dtype(as_native=as_native)
    elif as_native:
        default_dtype = tensorflow_as_native_dtype("bool")
    else:
        default_dtype = tensorflow_as_ivy_dtype_1("bool")
    return default_dtype
