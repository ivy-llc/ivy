# global
from typing import Optional, Union, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

# local
import ivy
from paddle.fluid.libpaddle import Place

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
