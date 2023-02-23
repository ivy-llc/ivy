from typing import Sequence
from typing import Union, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_math_ops

from ivy.func_wrapper import with_supported_dtypes
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
    original_bins = tf.identity(bins)
    flag_lower_interval = False
    if not extend_lower_interval:
        if tf.reduce_min(a) < bins[0]:
            bins = tf.concat([[tf.reduce_min(a)], bins], 0)
            flag_lower_interval = True
    flag_upper_interval = False
    if not extend_upper_interval:
        if tf.reduce_max(a) > bins[-1]:
            bins = tf.concat([bins, [tf.reduce_max(a)]], 0)
            flag_upper_interval = True
    ret = tfp.stats.histogram(
        x=a,
        edges=bins,
        axis=axis,
        weights=weights,
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=dtype,
        name="histogram",
    )
    # TODO: must delete the first element of the correct axis, this only works with
    #       1D input, usar take
    if not extend_lower_interval:
        if flag_lower_interval:
            ret = ret[1:]
    if not extend_upper_interval:
        if flag_upper_interval:
            ret = ret[:-1]
    # TODO: Tensorflow dtype argument is not working
    if dtype:
        ret = tf.cast(ret, dtype)
        original_bins = tf.cast(original_bins, dtype)
    return ret, original_bins


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
    np_math_ops.enable_numpy_methods_on_tensor()
    return tf.experimental.numpy.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype)


@with_supported_dtypes(
    {
        "2.9.1 and below": (
            "int32",
            "int64",
        )
    },
    backend_version,
)
def unravel_index(
    indices: Union[tf.Tensor, tf.Variable],
    shape: Tuple[int],
    /,
    *,
    out: Optional[Tuple[Union[tf.Tensor, tf.Variable]]] = None,
) -> Tuple:
    temp = indices
    output = []
    for dim in reversed(shape):
        output.append(temp % dim)
        temp = temp // dim
    ret = tf.constant(reversed(output), dtype=tf.int32)
    return tuple(ret)


def quantile(
    a: Union[tf.Tensor, tf.Variable],
    q: Union[tf.Tensor, float],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: Optional[str] = "linear",
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis

    result = tfp.stats.percentile(
        a, q * 100, axis=axis, interpolation=interpolation, keepdims=keepdims
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


def nanmedian(
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
