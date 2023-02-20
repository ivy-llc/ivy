# global
from typing import Optional, Union, Tuple, Sequence
import paddle
from ivy.exceptions import IvyNotImplementedException

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def median(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nanmean(
    a: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def quantile(
    a: paddle.Tensor,
    q: Union[paddle.Tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    raise IvyNotImplementedException()


def corrcoef(
    x: paddle.Tensor,
    /,
    *,
    y: Optional[paddle.Tensor] = None,
    rowvar: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nanmedian(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def unravel_index(
    indices: paddle.Tensor,
    shape: Tuple[int],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
