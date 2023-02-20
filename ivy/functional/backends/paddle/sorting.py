# global
import paddle
from typing import Optional

# local
import ivy

from . import backend_version
from ivy.exceptions import IvyNotImplementedException


def argsort(
    x: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def sort(
    x: paddle.Tensor,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def searchsorted(
    x: paddle.Tensor,
    v: paddle.Tensor,
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=paddle.int64,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
