# global
import math
from numbers import Number
from typing import Union, Optional, Tuple, List, Sequence, Iterable

import paddle

# local
import ivy
from ivy.exceptions import IvyNotImplementedException

# noinspection PyProtectedMember
from ivy.functional.ivy.manipulation import _calculate_out_shape
from . import backend_version


# Array API Standard #
# -------------------#


def concat(
    xs: Union[Tuple[paddle.Tensor, ...], List[paddle.Tensor]],
    /,
    *,
    axis: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def expand_dims(
    x: paddle.Tensor,
    /,
    *,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def flip(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def permute_dims(
    x: paddle.Tensor,
    /,
    axes: Tuple[int, ...],
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def reshape(
    x: paddle.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: Optional[str] = "C",
    allowzero: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def roll(
    x: paddle.Tensor,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def squeeze(
    x: paddle.Tensor,
    /,
    axis: Union[int, Sequence[int]],
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def stack(
    arrays: Union[Tuple[paddle.Tensor], List[paddle.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


# Extra #
# ------#


def split(
    x: paddle.Tensor,
    /,
    *,
    num_or_size_splits: Optional[Union[int, List[int]]] = None,
    axis: Optional[int] = 0,
    with_remainder: Optional[bool] = False,
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def repeat(
    x: paddle.Tensor,
    /,
    repeats: Union[int, Iterable[int]],
    *,
    axis: int = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def tile(
    x: paddle.Tensor, /, repeats: Sequence[int], *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if isinstance(repeats, paddle.Tensor):
        repeats = repeats.detach().cpu().numpy().tolist()
    return x.repeat(repeats)


def constant_pad(
    x: paddle.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def zero_pad(
    x: paddle.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def swapaxes(
    x: paddle.Tensor, axis0: int, axis1: int, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def clip(
    x: paddle.Tensor,
    x_min: Union[Number, paddle.Tensor],
    x_max: Union[Number, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def unstack(
    x: paddle.Tensor, /, *, axis: int = 0, keepdims: bool = False
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()
