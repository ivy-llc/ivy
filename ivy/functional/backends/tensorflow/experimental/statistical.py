from typing import Union, Optional, Tuple, Sequence
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_math_ops
import ivy


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
    min_a = tf.reduce_min(a)
    max_a = tf.reduce_max(a)
    if isinstance(bins, tf.Tensor) and range:
        raise ivy.exceptions.IvyException(
            "Must choose between specifying bins and range or bin edges directly"
        )
    if range:
        if isinstance(bins, int):
            bins = tf.cast(
                tf.linspace(start=range[0], stop=range[1], num=bins + 1), dtype=a.dtype
            )
    elif isinstance(bins, int):
        range = (min_a, max_a)
        bins = tf.cast(
            tf.linspace(start=range[0], stop=range[1], num=bins + 1), dtype=a.dtype
        )
    if tf.shape(bins)[0] < 2:
        raise ivy.exceptions.IvyException("bins must have at least 1 bin (size > 1)")
    if min_a < bins[0] and not extend_lower_interval:
        raise ivy.exceptions.IvyException(
            "Values of x outside of the intervals cause errors in tensorflow backend. "
            "Consider using extend_lower_interval to deal with this."
        )
    if max_a > bins[-1] and not extend_upper_interval:
        raise ivy.exceptions.IvyException(
            "Values of x outside of the intervals cause errors in tensorflow backend. "
            "Consider using extend_upper_interval to deal with this."
        )
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
    if density:
        pass
    # TODO: Tensorflow native dtype argument is not working
    if dtype:
        ret = tf.cast(ret, dtype)
        bins = tf.cast(bins, dtype)
    # TODO: weird error when returning bins: return ret, bins
    return ret


from ivy import with_supported_dtypes
from .. import backend_version


@with_supported_dtypes(
    {"2.9.1 and below": ("float",)},
    backend_version,
)
def median(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
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
    keepdims: bool = False,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    np_math_ops.enable_numpy_methods_on_tensor()
    return tf.experimental.numpy.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype)


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

    result = tfp.stats.percentile(
        a,
        tf.math.multiply(q, 100),
        axis=axis,
        interpolation=interpolation,
        keepdims=keepdims,
    )
    return result


def corrcoef(
    x: tf.Tensor,
    /,
    *,
    y: tf.Tensor,
    rowvar: bool = True,
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
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if overwrite_input:
        copied_input = tf.identity(input)
        return tfp.stats.percentile(
            copied_input,
            50.0,
            axis=axis,
            interpolation="midpoint",
            keepdims=keepdims,
        )
    return tfp.stats.percentile(
        input,
        50.0,
        axis=axis,
        interpolation="midpoint",
        keepdims=keepdims,
    )


def bincount(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    weights: Optional[Union[tf.Tensor, tf.Variable]] = None,
    minlength: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if weights is not None:
        ret = tf.math.bincount(x, weights=weights, minlength=minlength)
        ret = tf.cast(ret, weights.dtype)
    else:
        ret = tf.math.bincount(x, minlength=minlength)
        ret = tf.cast(ret, x.dtype)
    return ret
