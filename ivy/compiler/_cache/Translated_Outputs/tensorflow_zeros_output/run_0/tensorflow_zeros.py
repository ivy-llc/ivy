import tensorflow
import tensorflow as tf

from typing import Union
from typing import Sequence
from typing import Optional

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion
from .tensorflow__helpers import tensorflow_infer_dtype


@tensorflow_infer_dtype
@tensorflow_handle_array_like_without_promotion
def tensorflow_zeros(
    shape: Union[tf.TensorShape, Sequence[int]],
    *,
    dtype: tensorflow.DType,
    device: Optional[str] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    return tensorflow.zeros(shape, dtype=tensorflow.float32)
