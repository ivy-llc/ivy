import paddle
import paddle.nn.functional as F
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from typing import Optional, Tuple
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


# TODO: add support for the rest of the dtypes
# use numpy implementation with ivy functions
@with_unsupported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "float16",
                "complex",
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
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    if x.dtype not in [paddle.float32, paddle.float64]:
        x, mean, variance, scale, offset = [
            t.cast("float32") for t in [x, mean, variance, scale, offset]
        ]
    runningmean = mean
    runningvariance = variance
    data_formats = ["NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC", "NDHWC"]

    try:
        data_format = (
            data_formats[4:][x.ndim - 3]
            if data_format[-1] == "C"
            else data_formats[0:4][x.ndim - 2]
        )
    except IndexError:
        raise IndexError(
            "data_format must be one of 'NC', 'NCL', 'NCHW', 'NCDHW', "
            "'NLC', 'NHWC', 'NDHWC' but receive {}".format(data_format)
        )

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
        data_format=data_format,
    ).cast(x.dtype)
    return xnormalized, runningmean, runningvariance


batch_norm.partial_mixed_handler = lambda x, *args, scale, offset, **kwargs: (
    (x.ndim > 1 and x.ndim < 6)
    and (scale is None or scale.ndim == 1)
    and (offset is None or offset.ndim == 1)
)


def l1_normalize(
    x: paddle.Tensor, /, *, axis: int = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    if axis is None:
        axis = list(range(x.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    # Compute the L1 norm along the given axis
    norm = paddle.norm(x, p=1, axis=axis, keepdim=True)

    # Divide x by the L1 norm to obtain the normalized array
    norm = paddle.where(norm == 0, paddle.to_tensor([1], dtype=x.dtype), norm)
    if out is None:
        return x / norm
    else:
        out[:] = x / norm
        return out


def l2_normalize(
    x: paddle.Tensor, /, *, axis: int = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def instance_norm(
    x: paddle.Tensor,
    mean: paddle.Tensor,
    variance: paddle.Tensor,
    /,
    *,
    scale: Optional[paddle.Tensor] = None,
    offset: Optional[paddle.Tensor] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[
        Tuple[
            paddle.Tensor,
            paddle.Tensor,
            paddle.Tensor,
        ]
    ] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor,]:
    raise IvyNotImplementedException()


def lp_normalize(
    x: paddle.Tensor, /, *, p: float = 2, axis: int = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
