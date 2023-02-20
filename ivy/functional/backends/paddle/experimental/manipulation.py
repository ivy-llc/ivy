from typing import Optional, Union, Sequence, Tuple, NamedTuple, List
from numbers import Number
from .. import backend_version
import paddle
from ivy.exceptions import IvyNotImplementedException
import ivy


def moveaxis(
    a: paddle.Tensor,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def heaviside(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def flipud(
    m: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def vstack(
    arrays: Sequence[paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def hstack(
    arrays: Sequence[paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def rot90(
    m: paddle.Tensor,
    /,
    *,
    k: Optional[int] = 1,
    axes: Optional[Tuple[int, int]] = (0, 1),
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def top_k(
    x: paddle.Tensor,
    k: int,
    /,
    *,
    axis: Optional[int] = -1,
    largest: Optional[bool] = True,
    out: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def fliplr(
    m: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def i0(
    x: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def flatten(
    x: paddle.Tensor,
    /,
    *,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    order: Optional[str] = "C",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def vsplit(
    ary: paddle.Tensor,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def dsplit(
    ary: paddle.Tensor,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def atleast_1d(*arys: paddle.Tensor) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def dstack(
    arrays: Sequence[paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def atleast_2d(*arys: paddle.Tensor) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def atleast_3d(*arys: Union[paddle.Tensor, bool, Number]) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def take_along_axis(
    arr: paddle.Tensor,
    indices: paddle.Tensor,
    axis: int,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def hsplit(
    ary: paddle.Tensor,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def broadcast_shapes(shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    raise IvyNotImplementedException()
