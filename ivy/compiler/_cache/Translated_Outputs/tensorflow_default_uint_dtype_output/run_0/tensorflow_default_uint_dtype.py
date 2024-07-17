import tensorflow
import tensorflow as tf
import numpy as np

import math
from typing import Optional
from numbers import Number
from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default
from .tensorflow__helpers import tensorflow_default_dtype
from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_exists
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_is_native_array
from .tensorflow__helpers import tensorflow_is_uint_dtype
from .tensorflow__helpers import tensorflow_nested_argwhere

default_uint_dtype_stack = []
backend = ""


def tensorflow_default_uint_dtype(
    *,
    input: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    uint_dtype: Optional[Union[str, tf.DType]] = None,
    as_native: bool = False,
):
    global default_uint_dtype_stack
    if tensorflow_exists(uint_dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(uint_dtype)
        return str(tensorflow_as_ivy_dtype_1(uint_dtype))
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(input):
        if tensorflow_is_array(input):
            ret = tensorflow_dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):

            def is_native(x):
                return tensorflow_is_native_array(x)

            if tensorflow_nested_argwhere(
                input,
                lambda x: tensorflow_dtype(x) == "uint64"
                if is_native(x)
                else x > 9223372036854775807 and x != math.inf,
                stop_after_n_found=1,
            ):
                ret = tf.uint64
            elif default_uint_dtype_stack:
                ret = default_uint_dtype_stack[-1]
            else:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_uint_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "uint32"
        elif isinstance(input, Number):
            if input > 4294967295 and input != math.inf and backend != "torch":
                ret = tf.uint64
            elif default_uint_dtype_stack:
                ret = default_uint_dtype_stack[-1]
            else:
                def_dtype = tensorflow_default_dtype()
                if tensorflow_is_uint_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = "uint32"
    elif default_uint_dtype_stack:
        ret = default_uint_dtype_stack[-1]
    else:
        def_dtype = tensorflow_default_dtype()
        if tensorflow_is_uint_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = "uint32"
    if as_native:
        return tensorflow_as_native_dtype(ret)
    return str(tensorflow_as_ivy_dtype_1(ret))
