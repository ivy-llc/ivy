import tensorflow

from typing import Tuple
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_batch_norm(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    mean: Union[tensorflow.Tensor, tensorflow.Variable],
    variance: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    scale: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
    offset: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-05,
    momentum: Optional[float] = 0.1,
    data_format: Optional[str] = "NSC",
    out: Optional[
        Tuple[
            Union[tensorflow.Tensor, tensorflow.Variable],
            Union[tensorflow.Tensor, tensorflow.Variable],
            Union[tensorflow.Tensor, tensorflow.Variable],
        ]
    ] = None,
):
    xdims = len(x.shape)
    if data_format == "NCS":
        x = tensorflow.transpose(x, perm=(0, *range(2, xdims), 1))
    runningmean = mean
    runningvariance = variance
    if training:
        n = (
            tensorflow.size(x)
            if xdims == 1
            else tensorflow.divide(tensorflow.size(x), tensorflow.shape(x)[-1])
        )
        n = tensorflow.cast(n, x.dtype) if n.dtype != x.dtype else n
        dims = 0, *range(1, xdims - 1)
        mean = tensorflow.math.reduce_mean(x, axis=dims)
        variance = tensorflow.math.reduce_variance(x, axis=dims)
        runningmean = (1 - momentum) * runningmean + momentum * mean
        runningvariance = (1 - momentum) * runningvariance + momentum * variance * n / (
            n - 1
        )
    inv = 1.0 / tensorflow.math.sqrt(variance + eps)
    offset = 0 if offset is None else offset
    if scale is not None:
        inv = tensorflow.math.multiply(inv, scale)
    xnormalized = tensorflow.math.add(tensorflow.math.multiply(x, inv), offset)
    xnormalized = tensorflow.math.subtract(
        xnormalized, tensorflow.math.multiply(mean, inv)
    )
    if data_format == "NCS":
        xnormalized = tensorflow.transpose(
            xnormalized, perm=(0, xdims - 1, *range(1, xdims - 1))
        )
    return xnormalized, runningmean, runningvariance
