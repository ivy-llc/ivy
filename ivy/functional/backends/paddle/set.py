# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple

# local

from . import backend_version
from ivy.exceptions import IvyNotImplementedException


def unique_all(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_counts(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_inverse(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
