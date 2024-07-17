import tensorflow

from typing import Optional
from typing import Union
from typing import Sequence

from .tensorflow__helpers import tensorflow__infer_dtype
from .tensorflow__helpers import tensorflow_as_native_dtype


def tensorflow_prod(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[tensorflow.DType] = None,
    keepdims: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dtype = tensorflow_as_native_dtype(dtype)
    if dtype is None:
        dtype = tensorflow__infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return tensorflow.experimental.numpy.prod(
        x, axis=axis, dtype=dtype, keepdims=keepdims
    )
