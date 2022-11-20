from typing import Union, Optional, Tuple
import tensorflow as tf
import tensorflow_probability as tfp

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def median(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tfp.stats.percentile(
        input,
        50.0,
        axis=axis,
        interpolation="midpoint",
        keepdims=keepdims,
    )


def nanmean(
    a: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype)


@with_unsupported_dtypes({"2.9.1 and below": ("int8", "int16")}, backend_version)
def unravel_index(
    indices: Union[tf.Tensor, tf.Variable],
    shape: Tuple[int],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.unravel_index(indices, shape)
    return [tf.constant(ret[i]) for i in range(0, len(ret))]
