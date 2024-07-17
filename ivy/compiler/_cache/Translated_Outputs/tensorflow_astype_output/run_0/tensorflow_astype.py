import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_as_native_dtype


def tensorflow_astype(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    dtype: Union[tf.DType, str],
    /,
    *,
    copy: bool = True,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dtype = tensorflow_as_native_dtype(dtype)
    if x.dtype == dtype:
        return tensorflow.experimental.numpy.copy(x) if copy else x
    return tensorflow.cast(x, dtype)
