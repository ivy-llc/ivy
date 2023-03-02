from typing import Optional, Union

# global
import paddle
import paddle.nn.functional as F
import ivy
from ivy.utils.exceptions import IvyNotImplementedException

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _dtype_helper(x: paddle.Tensor):
    # used since all paddle activations only support float32 and float64
    if x.dtype not in [paddle.float32, paddle.float64]:
        return x.cast(paddle.float32), x.dtype
    return x, x.dtype


def logit(x: paddle.Tensor, /, *, eps: Optional[float] = None, out=None):
    x , x_dtype = _dtype_helper(x)

    if eps is None:
        nan = paddle.to_tensor([float('nan')]).cast(x.dtype)
        x = paddle.where(paddle.logical_or(x > 1, x < 0), nan, x)
    else:
        x = paddle.clip(x, eps, 1 - eps)
    return paddle.log(x / (1 - x)).cast(x_dtype)


def thresholded_relu(
    x: paddle.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)

    return F.thresholded_relu(x, threshold=threshold).cast(x_dtype)


def relu6(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)

    return F.relu6(x).cast(x_dtype)
