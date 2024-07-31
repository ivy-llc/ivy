import tensorflow

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_default_dtype_bknd
from .tensorflow__helpers import tensorflow_is_array_bknd
from .tensorflow__helpers import tensorflow_multiply


def tensorflow_add(
    x1: Union[float, tensorflow.Tensor, tensorflow.Variable],
    x2: Union[float, tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
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
    if x1.dtype.is_bool and x2.dtype.is_bool:
        return tensorflow.math.logical_or(x1, x2)
    if alpha not in (1, None):
        x2 = tensorflow_multiply(x2, alpha)
    return tensorflow.add(x1, x2)
