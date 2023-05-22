import paddle
import paddle.nn.functional as F
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from typing import Optional
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


# TODO: add support for the rest of the dtypes
# use numpy implementation with ivy functions
@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def batch_norm(
    x: paddle.Tensor,
    mean: paddle.Tensor,
    variance: paddle.Tensor,
    /,
    *,
    scale: Optional[paddle.Tensor] = None,
    offset: Optional[paddle.Tensor] = None,
    training: bool = False,
    eps: float = 1e-5,
    momentum: float = 1e-1,
    out: Optional[paddle.Tensor] = None,
):
    if x.dtype not in [paddle.float32, paddle.float64]:
        x, mean, variance, scale, offset = [
            t.cast("float32") for t in [x, mean, variance, scale, offset]
        ]
    runningmean = mean
    runningvariance = variance
    data_format = ["", "", "NC", "NLC", "NHWC", "NDHWC"]
    with ivy.ArrayMode(False):
        if training:
            x_shape = paddle.to_tensor(x.shape)
            x_size = paddle.prod(x_shape)
            n = (x_size if x.ndim == 1 else ivy.divide(x_size, x_shape[-1])).cast(
                x.dtype
            )
            dims = (0, *range(1, x.ndim - 1))
            mean = ivy.mean(x, axis=dims)
            variance = ivy.var(x, axis=dims)
            # runningmean = (1 - momentum) * runningmean + momentum * mean
            runningmean = ivy.add(
                ivy.multiply(ivy.subtract(1, momentum), runningmean),
                ivy.multiply(momentum, mean),
            )
            # runningvariance = (
            #    1 - momentum
            # ) * runningvariance + momentum * variance * n / (n - 1)
            runningvariance = ivy.add(
                ivy.multiply(ivy.subtract(1, momentum), runningvariance),
                ivy.divide(ivy.multiply(ivy.multiply(momentum, variance), n), n - 1),
            )

    xnormalized = F.batch_norm(
        x,
        running_mean=mean,
        running_var=variance,
        weight=scale,
        bias=offset,
        training=training,
        momentum=momentum,
        epsilon=eps,
        data_format=data_format[x.ndim],
    ).cast(x.dtype)
    return xnormalized, runningmean, runningvariance


def l2_normalize(
    x: paddle.Tensor, /, *, axis: int = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def instance_norm(
    x: paddle.Tensor,
    /,
    *,
    scale: Optional[paddle.Tensor],
    bias: Optional[paddle.Tensor],
    eps: float = 1e-05,
    momentum: Optional[float] = 0.1,
    data_format: str = "NCHW",
    running_mean: Optional[paddle.Tensor] = None,
    running_stddev: Optional[paddle.Tensor] = None,
    affine: Optional[bool] = True,
    track_running_stats: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def lp_normalize(
    x: paddle.Tensor, /, *, p: float = 2, axis: int = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
