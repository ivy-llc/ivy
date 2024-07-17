import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Sequence
from typing import Union

from .tensorflow__helpers import tensorflow__check_bounds_and_get_shape


def tensorflow_random_uniform(
    *,
    low: Union[float, tensorflow.Tensor, tensorflow.Variable] = 0.0,
    high: Union[float, tensorflow.Tensor, tensorflow.Variable] = 1.0,
    shape: Optional[Union[tf.TensorShape, Sequence[int], tensorflow.Tensor]] = None,
    dtype: tf.DType,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    shape = tensorflow__check_bounds_and_get_shape(low, high, shape)
    low = tensorflow.cast(low, dtype)
    high = tensorflow.cast(high, dtype)
    if seed:
        tensorflow.random.set_seed(seed)
    return tensorflow.random.uniform(shape, low, high, dtype=dtype, seed=seed)
