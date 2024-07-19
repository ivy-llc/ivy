import tensorflow

from typing import Union

from .tensorflow__helpers import tensorflow__is_variable_bknd
from .tensorflow__helpers import tensorflow_astype
from .tensorflow__helpers import tensorflow_is_array_bknd
from .tensorflow__helpers import tensorflow_is_ivy_array_bknd


def tensorflow_inplace_update(
    x: Union[tensorflow.Tensor, tensorflow.Tensor],
    val: Union[tensorflow.Tensor, tensorflow.Tensor],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
):
    if tensorflow_is_array_bknd(x) and tensorflow_is_array_bknd(val):
        if keep_input_dtype:
            val = tensorflow_astype(val, x.dtype)
        (x_native, val_native), _ = (x, val), "_"
        if tensorflow__is_variable_bknd(x_native):
            x_native.assign(val_native)
            if tensorflow_is_ivy_array_bknd(x):
                x = x_native
            else:
                x = tensorflow.convert_to_tensor(x_native)
        else:
            x = x_native
        return x
    else:
        return val
