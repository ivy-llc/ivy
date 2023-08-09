# global
from typing import Optional, Union, Sequence
import paddle

from ivy.functional.backends.paddle import backend_version
from ivy.utils.exceptions import IvyNotImplementedException

# local
import ivy
from paddle.device import core
from ivy.functional.ivy.random import (
    _check_shapes_broadcastable,
)
from ivy import with_unsupported_device_and_dtypes

# dirichlet


@with_unsupported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "int8",
                "int16",
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
def dirichlet(
    alpha: Union[paddle.Tensor, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
    seed: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    size = size if size is not None else len(alpha)
    dtype = dtype if dtype is not None else paddle.float64
    if seed is not None:
        paddle.seed(seed)
    res = paddle.to_tensor(
        paddle.distribution.Dirichlet(concentration=alpha).sample(shape=size),
        dtype=dtype,
    )
    return res


def beta(
    alpha: Union[float, paddle.Tensor],
    beta: Union[float, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[Union[paddle.dtype, ivy.Dtype]] = None,
    device: core.Place = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def gamma(
    alpha: Union[float, paddle.Tensor],
    beta: Union[float, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[Union[paddle.dtype, ivy.Dtype]] = None,
    device: core.Place = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "int8",
                "int16",
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
def poisson(
    lam: Union[float, paddle.Tensor],
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: core.Place,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    fill_value: Optional[Union[float, int]] = 0,
    out: Optional[paddle.Tensor] = None,
):
    lam = paddle.to_tensor(lam, dtype=dtype)
    if shape is None:
        if seed:
            paddle.seed(seed)
        return paddle.poisson(lam)
    if seed:
        paddle.seed(seed)
    _check_shapes_broadcastable(lam.shape, shape)
    shape = paddle.to_tensor(shape, dtype="int32")
    if not lam.shape:
        return paddle.poisson(lam)
    paddle.broadcast_to(lam, shape)
    return paddle.poisson(lam)


def bernoulli(
    probs: Union[float, paddle.Tensor],
    *,
    logits: Union[float, paddle.Tensor] = None,
    shape: Optional[Union[ivy.NativeArray, Sequence[int]]] = None,
    device: core.Place,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
