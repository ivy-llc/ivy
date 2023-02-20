# global
import math

import paddle
from ivy.exceptions import IvyNotImplementedException
from typing import Optional, Tuple

import ivy

from .. import backend_version

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size


def diagflat(
    x: paddle.Tensor,
    /,
    *,
    offset: Optional[int] = 0,
    padding_value: Optional[float] = 0,
    align: Optional[str] = "RIGHT_LEFT",
    num_rows: Optional[int] = -1,
    num_cols: Optional[int] = -1,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def kron(
    a: paddle.Tensor,
    b: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def matrix_exp(
    x: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def eig(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor]:
    raise IvyNotImplementedException()


def eigvals(x: paddle.Tensor, /) -> paddle.Tensor:
    raise IvyNotImplementedException()


def adjoint(
    x: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
