from typing import Sequence
from typing import Union, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


# TODO: Avoid error when inputs are out of range and extend_lower_interval or
#       extend_upper_interval are false.
#       Tensorflow native dtype argument is not working (casting was required)
@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
            "bfloat16",
            "float16",
        )
    },
    backend_version,
)
def histogram(
    a: tf.Tensor,
    /,
    *,
    bins: Optional[Union[int, tf.Tensor, str]] = None,
    axis: Optional[tf.Tensor] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[tf.DType] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[tf.Tensor] = None,
    density: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor]:
    if range:
        if type(bins) == int:
            bins = tf.cast(tf.linspace(start=range[0], stop=range[1], num=bins + 1), dtype=a.dtype)
    ret = tfp.stats.histogram(
        x=a,
        edges=bins,
        axis=axis,
        weights=weights,
        extend_upper_interval=extend_upper_interval,
        extend_lower_interval=extend_lower_interval,
        dtype=dtype,
        name="histogram",
    )
    ret = tf.cast(ret, dtype=dtype)
    bins = tf.cast(bins, dtype=dtype)
    return ret, bins


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


def quantile(
    a: Union[tf.Tensor, tf.Variable],
    q: Union[tf.Tensor, float],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis

    # In tensorflow, it requires percentile in range [0, 100], while in the other
    # backends the quantile has to be in range [0, 1].
    q = q * 100

    # The quantile instance method in other backends is equivalent of
    # percentile instance method in tensorflow_probability
    result = tfp.stats.percentile(
        a, q, axis=axis, interpolation=interpolation, keepdims=keepdims
    )
    return result


def corrcoef(
    x: tf.Tensor,
    /,
    *,
    y: tf.Tensor,
    rowvar: Optional[bool] = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> tf.Tensor:
    if y is None:
        xarr = x
    else:
        axis = 0 if rowvar else 1
        xarr = tf.concat([x, y], axis=axis)

    if rowvar:
        mean_t = tf.reduce_mean(xarr, axis=1, keepdims=True)
        cov_t = ((xarr - mean_t) @ tf.transpose(xarr - mean_t)) / (x.shape[1] - 1)
    else:
        mean_t = tf.reduce_mean(xarr, axis=0, keepdims=True)
        cov_t = (tf.transpose(xarr - mean_t) @ (xarr - mean_t)) / (x.shape[1] - 1)

    cov2_t = tf.linalg.diag(1 / tf.sqrt(tf.linalg.diag_part(cov_t)))
    cor = cov2_t @ cov_t @ cov2_t
    return cor
