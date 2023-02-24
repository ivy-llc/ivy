"""Collection of Paddle activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union

# global
import paddle
import paddle.nn.functional as F
# local
import ivy
from . import backend_version


def _dtype_helper(x: paddle.Tensor):
    # used since all paddle activations only support float32 and float64
    if x.dtype not in [paddle.float32, paddle.float64]:
        return x.cast(paddle.float32), x.dtype
    return x, x.dtype


def relu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)

    return F.relu(x).cast(x_dtype)


def leaky_relu(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)

    return F.leaky_relu(x, negative_slope=alpha).cast(x_dtype)


def gelu(
    x: paddle.Tensor,
    /,
    *,
    approximate: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)
    ret = F.gelu(x, approximate=approximate)
    if ivy.is_int_dtype(x_dtype):
        ret = ret.round()
    return ret.cast(x_dtype)


def sigmoid(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
            ) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)
    # ideally, the output should be cast to x_dtype, 
    # but since sigmoid result is bounded between 0 and 1
    # if x_dtype was an integer the result will be zeros

    return F.sigmoid(x)


def softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)

    return F.softmax(x, axis=axis).cast(x_dtype)


def softplus(
    x: paddle.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = 1,
    threshold: Optional[Union[int, float]] = 20,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)

    return F.softplus(x, beta=beta, threshold=threshold).cast(x_dtype)


def log_softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    out: Optional[paddle.Tensor] = None,
):
    x , x_dtype = _dtype_helper(x)

    return F.log_softmax(x, axis=axis).cast(x_dtype)


def mish(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    x , x_dtype = _dtype_helper(x)

    return F.mish(x).cast(x_dtype)
