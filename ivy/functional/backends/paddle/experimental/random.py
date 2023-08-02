# global
from typing import Optional, Union, Sequence
import paddle
from ivy.functional.backends.paddle.device import to_device
from ivy import with_unsupported_device_and_dtypes
from ivy.functional.backends.paddle import backend_version
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.functional.ivy.random import _check_bounds_and_get_shape

# local
import ivy
from paddle.device import core

# dirichlet


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
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


# beta
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
    if seed is not None:
        paddle.seed(seed)
    shape = _check_bounds_and_get_shape(alpha, beta, shape)
    dtype = paddle.float32 if dtype is None else dtype
    beta = paddle.cast(beta, alpha.dtype)
    dist = paddle.distribution.Beta(alpha, beta)
    sample = dist.sample(shape)
    sample = paddle.cast(sample, dtype)
    sample = to_device(sample, device) if device is not None else sample
    return sample


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
    raise IvyNotImplementedException()


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
