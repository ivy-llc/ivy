from numbers import Number
from typing import Optional, Tuple, Union

import paddle


import ivy
from ivy.exceptions import IvyNotImplementedException

from . import backend_version

# Array API Standard #
# ------------------ #


def argmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def argmin(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[paddle.dtype] = None,
    select_last_index: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nonzero(
    x: paddle.Tensor,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor]]:
    raise IvyNotImplementedException()


def where(
    condition: paddle.Tensor,
    x1: Union[float, int, paddle.Tensor],
    x2: Union[float, int, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


# Extra #
# ----- #


def argwhere(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    raise IvyNotImplementedException()
