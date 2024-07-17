import tensorflow
import tensorflow as tf
import numpy as np

from typing import Optional
from typing import Union
from numbers import Number

from .tensorflow__helpers import tensorflow__check_complex128
from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default
from .tensorflow__helpers import tensorflow_default_dtype
from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_exists
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_is_complex_dtype
from .tensorflow__helpers import tensorflow_nested_argwhere

default_complex_dtype_stack = []


def tensorflow_default_complex_dtype(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    complex_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_complex_dtype_stack
    if tensorflow_exists(complex_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(complex_dtype)
        return str(tensorflow_as_ivy_dtype_1(complex_dtype))
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(input):
        if tensorflow_is_array(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere(
                input, lambda x: tensorflow__check_complex128(x), stop_after_n_found=1
            ):
                ret = tf.complex128
            elif not default_complex_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_complex_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "complex64"
            else:
                ret = default_complex_dtype_stack[-1]
        elif isinstance(input, Number):
            if tensorflow__check_complex128(input):
                ret = tf.complex128
            elif not default_complex_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_complex_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "complex64"
            else:
                ret = default_complex_dtype_stack[-1]
    elif not default_complex_dtype_stack:
        def_dtype = tensorflow_default_dtype()
        if tensorflow_is_complex_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = "complex64"
    else:
        ret = default_complex_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype_1(ret))
