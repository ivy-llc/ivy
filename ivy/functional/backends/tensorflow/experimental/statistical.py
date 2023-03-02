from typing import Union, Optional, Tuple, Sequence
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

tensorfolow_histogram(
    x: Union[tf.Tensor, tf.Variable,
    edges: Union[ dtypex1-Dedges.shape[1:]xaxisrank(edges) > 1edges[k]edges.shape[1:]Tensorx],
    axis: [DTensor]=None,
    weights:Union[bin[],Tensordtypeshapexx[]]=None,
    extend_lower_interval:bool=False,
    extend_upper_interval:[bool]=False,
    dtype: Union[[int32],[int64] [value:x.dtype]=None],
    name:str[histogram]=None
)

    return tf.math.cumsum(x, edges, axes, reverse)

@staticmethod
def tensorfolow_histogram2(self: tf.Tensor, 
    /, 
    *,
    name: Optional[tf.Tensor],
    data:Optional[Union[tf.Tensor]],
    step:Optional[Union[tf.Tensor]] = None, 
    buckets:Optional[tf.Tensor[int]], 
    description:Optional[tf.Tensor[str]]
    ) -> ivy.Array:


    a = tf.summary.create_file_writer('test/logs')
    with a.as_default():
        for step in range(100):
        
        # Generate fake "activations".

            activations = [
                tf.random.normal([1000], mean=step, stddev=1),
                tf.random.normal([1000], mean=step, stddev=10),
                tf.random.normal([1000], mean=step, stddev=100),
            ]

            tf.summary.histogram("layer1/activate", activations[0], step=step)
            tf.summary.histogram("layer2/activate", activations[1], step=step)
            tf.summary.histogram("layer3/activate", activations[2], step=step)

    return using_file_hystogram