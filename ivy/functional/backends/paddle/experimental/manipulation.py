from typing import Optional, Union, Sequence, Tuple, NamedTuple, List
from numbers import Number
from .. import backend_version
from ivy.func_wrapper import with_unsupported_dtypes
import paddle
from ivy.utils.exceptions import IvyNotImplementedException
import ivy
from ivy.func_wrapper import with_unsupported_dtypes


@with_unsupported_dtypes(
    {"2.4.2 and below": ('int8', 'int16', 'uint8', 'uint16')},
    backend_version,
)
def moveaxis(
    a: paddle.Tensor,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.moveaxis(a, source, destination)


@with_unsupported_dtypes(
    {"2.4.2 and below": ('int8', 'int16', 'uint8', 'uint16', 'bfloat16',
                         'float16', 'complex64', 'complex128', 'bool')},
    backend_version,
)
def heaviside(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.heaviside(x1, x2)


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


@with_unsupported_dtypes(
    {"2.4.2 and below": ("uint16", "bfloat16", "complex64", "complex128", "bool")},
    backend_version,
)
def top_k(
    x: paddle.Tensor,
    k: int,
    /,
    *,
    axis: Optional[int] = -1,
    largest: Optional[bool] = True,
    out: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    topk_res = NamedTuple("top_k", [("values", paddle.Tensor), 
                                    ("indices", paddle.Tensor)])
    val, indices = paddle.topk(x, k, axis=axis, largest=largest)
    return topk_res(val, indices)
    

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
