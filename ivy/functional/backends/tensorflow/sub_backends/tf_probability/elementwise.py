from typing import Optional
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp


def trapz(
    y: Union[tf.Tensor, tf.Variable],
    /,
    *,
    x: Optional[Union[tf.Tensor, tf.Variable]] = None,
    dx: float = 1.0,
    axis: int = -1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tfp.math.trapz(y, x=x, dx=dx, axis=axis, name=None)
