import tensorflow

from typing import Union
from numbers import Number
from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_infer_dtype


@tensorflow_infer_dtype
@tensorflow_handle_array_like_without_promotion
def tensorflow_full_like(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    fill_value: Number,
    *,
    dtype: tensorflow.DType,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.experimental.numpy.full_like(x, fill_value, dtype=dtype)
