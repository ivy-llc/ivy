# global
from typing import Optional, Union, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_device_and_dtypes

# local
import ivy
from paddle.fluid.libpaddle import Place

from . import backend_version


# dirichlet
def dirichlet(
    alpha: Union[paddle.Tensor, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
    seed: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def beta(
    alpha: Union[float, paddle.Tensor],
    beta: Union[float, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: Optional[Union[paddle.dtype, ivy.Dtype]] = None,
    device: Place = None,
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
    device: Place = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def poisson(
    lam: Union[float, paddle.Tensor],
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Place,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def bernoulli(
    probs: Union[float, paddle.Tensor],
    *,
    logits: Union[float, paddle.Tensor] = None,
    shape: Optional[Union[ivy.NativeArray, Sequence[int]]] = None,
    device: Place,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("float16",)}},
    backend_version,
)
def laplace(
    loc: Union[paddle.Tensor, float, Sequence[float]],
    scale: Union[paddle.Tensor, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Place = None,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
    seed: Optional[int] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()

    # TODO: Uncomment after the next version of paddlepaddle is released with Laplace
    # if size is None:
    #     if isinstance(loc, float) and isinstance(scale, float):
    #         size = 1
    #     else:
    #         size = paddle.broadcast_shape(loc.shape, scale.shape)

    # if dtype is None:
    #     dtype = paddle.float64
    # else:
    #     dtype = dtype

    # if seed is not None:
    #     paddle.seed(seed)

    # TODO: import at the start of the file when this section is uncommented
    # from ivy.functional.backends.paddle.device import to_device

    # return to_device(
    #     paddle.cast(
    #         paddle.distribution.Laplace(loc=loc, scale=scale).sample(size),
    #         dtype=dtype,
    #     ),
    #     device
    # )
