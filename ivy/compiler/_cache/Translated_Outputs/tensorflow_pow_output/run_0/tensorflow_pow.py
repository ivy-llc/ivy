import tensorflow

import math
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_any
from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_default_dtype_bknd
from .tensorflow__helpers import tensorflow_is_array_bknd
from .tensorflow__helpers import tensorflow_is_complex_dtype_bknd
from .tensorflow__helpers import tensorflow_is_int_dtype_bknd
from .tensorflow__helpers import tensorflow_isinf


def tensorflow_pow(
    x1: Union[tensorflow.Tensor, tensorflow.Variable],
    x2: Union[int, float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if tensorflow_is_complex_dtype_bknd(x1) and tensorflow_any(tensorflow_isinf(x2)):
        ret = tensorflow.experimental.numpy.power(x1, x2)
        return tensorflow.where(
            tensorflow_isinf(x2),
            math.nan + math.nan * 1.0j if x2 < 0 else -0 * 1.0j,
            ret,
        )
    if tensorflow_is_complex_dtype_bknd(x2) and tensorflow_any(x1 == 0):
        ret = tensorflow.experimental.numpy.power(x1, x2)
        return tensorflow.where(x1 == 0, math.nan + math.nan * 1.0j, ret)
    oirg_x1 = x1
    oirg_x2 = x2
    try:
        dtype = (
            x1.dtype
            if hasattr(x1, "dtype")
            else x2.dtype if hasattr(x2, "dtype") else tensorflow_default_dtype_bknd()
        )
        if not tensorflow_is_array_bknd(x1):
            x1 = tensorflow_asarray(x1, dtype=dtype)
        if not tensorflow_is_array_bknd(x2):
            x2 = tensorflow_asarray(x2, dtype=dtype)
    except:
        x1 = oirg_x1
        x2 = oirg_x2
    if tensorflow_is_int_dtype_bknd(x1) and tensorflow_any(x2 < 0):
        return tensorflow.cast(
            tensorflow.experimental.numpy.power(
                tensorflow.cast(x1, tensorflow.float32), x2
            ),
            x1.dtype,
        )
    return tensorflow.experimental.numpy.power(x1, x2)
