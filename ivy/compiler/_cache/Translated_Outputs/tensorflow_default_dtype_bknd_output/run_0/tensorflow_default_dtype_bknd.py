import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default_bknd
from .tensorflow__helpers import tensorflow_default_complex_dtype_bknd
from .tensorflow__helpers import tensorflow_default_float_dtype_bknd
from .tensorflow__helpers import tensorflow_default_int_dtype_bknd
from .tensorflow__helpers import tensorflow_exists_bknd
from .tensorflow__helpers import tensorflow_is_complex_dtype_bknd
from .tensorflow__helpers import tensorflow_is_float_dtype_bknd
from .tensorflow__helpers import tensorflow_is_int_dtype_bknd
from .tensorflow__helpers import tensorflow_is_uint_dtype_bknd

default_dtype_stack = []
default_float_dtype_stack = []
default_float_dtype_stack = []
default_dtype_stack = []


def tensorflow_default_dtype_bknd(
    *,
    dtype: Optional[Union[str, str]] = None,
    item: Optional[Union[tensorflow.Tensor, tf.Tensor]] = None,
    as_native: bool = False,
):
    if tensorflow_exists_bknd(dtype):
        if as_native is True:
            return tensorflow_as_native_dtype(dtype)
        return tensorflow_as_ivy_dtype(dtype)
    as_native = tensorflow_default_bknd(as_native, False)
    if tensorflow_exists_bknd(item):
        if hasattr(item, "override_dtype_check"):
            return item.override_dtype_check()
        elif isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif tensorflow_is_complex_dtype_bknd(item):
            return tensorflow_default_complex_dtype_bknd(
                input=item, as_native=as_native
            )
        elif tensorflow_is_float_dtype_bknd(item):
            return tensorflow_default_float_dtype_bknd(input=item, as_native=as_native)
        elif tensorflow_is_uint_dtype_bknd(item):
            return tensorflow_default_int_dtype_bknd(input=item, as_native=as_native)
        elif tensorflow_is_int_dtype_bknd(item):
            return tensorflow_default_int_dtype_bknd(input=item, as_native=as_native)
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
    return tensorflow_as_ivy_dtype(ret)
