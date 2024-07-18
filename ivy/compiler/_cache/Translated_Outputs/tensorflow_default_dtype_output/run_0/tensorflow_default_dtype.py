import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default
from .tensorflow__helpers import tensorflow_default_complex_dtype
from .tensorflow__helpers import tensorflow_default_float_dtype
from .tensorflow__helpers import tensorflow_default_int_dtype
from .tensorflow__helpers import tensorflow_exists
from .tensorflow__helpers import tensorflow_is_complex_dtype
from .tensorflow__helpers import tensorflow_is_float_dtype
from .tensorflow__helpers import tensorflow_is_int_dtype
from .tensorflow__helpers import tensorflow_is_uint_dtype

default_dtype_stack = []
default_float_dtype_stack = []


def tensorflow_default_dtype(
    *,
    dtype: Optional[Union[str, str]] = None,
    item: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    as_native: bool = False,
):
    if tensorflow_exists(dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(dtype)
        return tensorflow_as_ivy_dtype_1(dtype)
    as_native = tensorflow_default(as_native, False)
    if tensorflow_exists(item):
        if hasattr(item, "override_dtype_check"):
            return item.override_dtype_check()
        elif isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif tensorflow_is_complex_dtype(item):
            return tensorflow_default_complex_dtype(input=item, as_native=as_native)
        elif tensorflow_is_float_dtype(item):
            return tensorflow_default_float_dtype(input=item, as_native=as_native)
        elif tensorflow_is_uint_dtype(item):
            return tensorflow_default_int_dtype(input=item, as_native=as_native)
        elif tensorflow_is_int_dtype(item):
            return tensorflow_default_int_dtype(input=item, as_native=as_native)
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
    return tensorflow_as_ivy_dtype_1(ret)
