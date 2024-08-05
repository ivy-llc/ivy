import tensorflow
import tensorflow as tf

from typing import Sequence
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow__check_bounds_and_get_shape_bknd
from .tensorflow__helpers import tensorflow_infer_dtype


@tensorflow_infer_dtype
def tensorflow_random_uniform(
    *,
    low: Union[float, tensorflow.Tensor, tensorflow.Variable] = 0.0,
    high: Union[float, tensorflow.Tensor, tensorflow.Variable, None] = 1.0,
    shape: Optional[Union[tf.TensorShape, Sequence[int], tensorflow.Tensor]] = None,
    dtype: tf.DType,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    shape = tensorflow__check_bounds_and_get_shape_bknd(
        low,
        float(
            tensorflow.experimental.numpy.finfo(tensorflow.float32).max
            if dtype is None
            else tensorflow.experimental.numpy.finfo(dtype).max
        )
        if high is None
        else high,
        shape,
    )
    low = tensorflow.cast(low, dtype)
    if high is not None:
        high = tensorflow.cast(high, dtype)
    if seed:
        tensorflow.random.set_seed(seed)
    return tensorflow.random.uniform(shape, low, high, dtype=dtype, seed=seed)
