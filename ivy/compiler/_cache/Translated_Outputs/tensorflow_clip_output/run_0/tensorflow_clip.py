import tensorflow

from typing import Union
from numbers import Number
from typing import Optional

from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_asarray
from .tensorflow__helpers import tensorflow_promote_types_bknd


def tensorflow_clip(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    x_min: Optional[Union[Number, tensorflow.Tensor, tensorflow.Variable]] = None,
    x_max: Optional[Union[Number, tensorflow.Tensor, tensorflow.Variable]] = None,
    *,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if x_min is None and x_max is None:
        raise ValueError("At least one of the x_min or x_max must be provided")
    promoted_type = x.dtype
    if x_min is not None:
        if not hasattr(x_min, "dtype"):
            x_min = tensorflow_asarray(x_min)
        promoted_type = tensorflow_as_native_dtype(
            tensorflow_promote_types_bknd(x.dtype, x_min.dtype)
        )
    if x_max is not None:
        if not hasattr(x_max, "dtype"):
            x_max = tensorflow_asarray(x_max)
        promoted_type = tensorflow_as_native_dtype(
            tensorflow_promote_types_bknd(promoted_type, x_max.dtype)
        )
        x_max = tensorflow.cast(x_max, promoted_type)
    x = tensorflow.cast(x, promoted_type)
    if x_min is not None:
        x_min = tensorflow.cast(x_min, promoted_type)
    cond = True
    if x_min is not None and x_max is not None:
        if tensorflow.math.reduce_any(
            tensorflow.experimental.numpy.greater(x_min, x_max)
        ):
            cond = False
    if cond:
        return tensorflow.experimental.numpy.clip(x, x_min, x_max)
    else:
        return tensorflow.experimental.numpy.minimum(
            x_max, tensorflow.experimental.numpy.maximum(x, x_min)
        )
