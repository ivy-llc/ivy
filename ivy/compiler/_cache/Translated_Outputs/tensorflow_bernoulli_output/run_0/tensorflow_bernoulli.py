import tensorflow
import tensorflow as tf

from typing import Optional
from typing import Union
from typing import Sequence

from .tensorflow__helpers import tensorflow__check_shapes_broadcastable_bknd
from .tensorflow__helpers import tensorflow_infer_dtype


@tensorflow_infer_dtype
def tensorflow_bernoulli(
    probs: Union[float, tensorflow.Tensor, tensorflow.Variable],
    *,
    logits: Union[float, tensorflow.Tensor, tensorflow.Variable] = None,
    shape: Optional[Union[tf.TensorShape, Sequence[int]]] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    seed: Optional[int] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dtype = dtype if dtype is not None else probs.dtype
    if logits is not None:
        probs = tensorflow.nn.softmax(logits, -1)
    if not tensorflow__check_shapes_broadcastable_bknd(shape, probs.shape):
        shape = probs.shape
    return tensorflow.keras.backend.random_bernoulli(shape, probs, dtype, seed)
