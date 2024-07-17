import tensorflow
import tensorflow as tf
import numpy as np

import math
from typing import Optional
from typing import Union
from numbers import Number

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default
from .tensorflow__helpers import tensorflow_default_dtype
from .tensorflow__helpers import tensorflow_default_int_dtype
from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_exists
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_is_int_dtype
from .tensorflow__helpers import tensorflow_nested_argwhere

default_int_dtype_stack = []
backend = ""


def tensorflow_default_int_dtype(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    int_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_int_dtype_stack
    if tensorflow_exists(int_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(int_dtype)
        return str(tensorflow_as_ivy_dtype_1(int_dtype))
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(input):
        if tensorflow_is_array(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, tuple):
            ret = tensorflow_default_int_dtype()
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if tensorflow_nested_argwhere(
                input,
                lambda x: tensorflow_dtype(x) == "uint64"
                if tensorflow_is_array(x)
                else x > 9223372036854775807 and x != math.inf,
                stop_after_n_found=1,
            ):
                ret = tf.uint64
            elif tensorflow_nested_argwhere(
                input,
                lambda x: tensorflow_dtype(x) == "int64"
                if tensorflow_is_array(x)
                else x > 2147483647 and x != math.inf,
                stop_after_n_found=1,
            ):
                ret = tf.int64
            elif not default_int_dtype_stack:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_int_dtype(def_dtype):
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
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "int32"
            else:
                ret = default_int_dtype_stack[-1]
    elif not default_int_dtype_stack:
        def_dtype = tensorflow_default_dtype()
        if tensorflow_is_int_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = "int32"
    else:
        ret = default_int_dtype_stack[-1]
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype_1(ret))
