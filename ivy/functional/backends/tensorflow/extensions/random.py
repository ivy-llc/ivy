from typing import Union, Optional, Sequence
import tensorflow as tf
import ivy
from .. import backend_version
from tensorflow_probability import distributions as tfd

# local
from ivy.func_wrapper import with_unsupported_dtypes


# dirichlet
@with_unsupported_dtypes({"2.9.1 and below": ("blfoat16", "float16")}, backend_version)
def dirichlet(
    alpha: Union[tf.Tensor, tf.Variable, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    seed: Optional[int] = None,
    dtype: Optional[tf.Tensor] = None,
) -> Union[tf.Tensor, tf.Variable]:
    size = size if size is not None else len(alpha)

    if dtype is None:
        dtype = tf.float64
    else:
        dtype = dtype
    if seed is not None:
        tf.random.set_seed(seed)
    return tf.cast(
        tfd.Dirichlet(
            concentration=alpha,
            validate_args=False,
            allow_nan_stats=True,
            force_probs_to_zero_outside_support=False,
            name="Dirichlet",
        ).sample(size),
        dtype=dtype,
    )
