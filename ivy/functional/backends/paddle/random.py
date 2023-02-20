"""Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature."""

# global
import paddle
from typing import Optional, Union, Sequence

# local
import ivy
from ivy.exceptions import IvyNotImplementedException
from . import backend_version
from paddle.fluid.libpaddle import Place
# Extra #
# ------#


def random_uniform(
    *,
    low: Union[float, paddle.Tensor] = 0.0,
    high: Union[float, paddle.Tensor] = 1.0,
    shape: Optional[Union[paddle.Tensor, ivy.NativeShape, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: Place,
    seed=None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def random_normal(
    *,
    mean: Union[float, paddle.Tensor] = 0.0,
    std: Union[float, paddle.Tensor] = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[paddle.Tensor] = None,
    replace: bool = True,
    device: Place,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def randint(
    low: Union[int, paddle.Tensor],
    high: Union[int, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    device: Place,
    dtype: Optional[Union[paddle.dtype, ivy.Dtype]] = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def seed(*, seed_value: int = 0) -> None:
    raise IvyNotImplementedException()


def shuffle(
    x: paddle.Tensor,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
