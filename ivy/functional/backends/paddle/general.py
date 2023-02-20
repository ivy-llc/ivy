"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""
# global
from functools import reduce
from numbers import Number
from operator import mul
from typing import Optional, Union, Sequence, Callable, List
import paddle
import numpy as np
# local
import ivy
from ivy.exceptions import IvyNotImplementedException

from . import backend_version


def is_native_array(x, /, *, exclusive=False):
    raise IvyNotImplementedException()


def array_equal(x0: paddle.Tensor, x1: paddle.Tensor, /) -> bool:
    raise IvyNotImplementedException()


def container_types():
    return []


def current_backend_str() -> str:
    return "paddle"


def get_item(
    x: paddle.Tensor,
    query: paddle.Tensor,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def to_numpy(
    x: Union[paddle.Tensor, List[paddle.Tensor]], /, *, copy: bool = True
) -> Union[np.ndarray, List[np.ndarray]]:
    raise IvyNotImplementedException()


def to_scalar(x: paddle.Tensor, /) -> Number:
    if isinstance(x, (float, int)):
        return x
    return x.item()


def to_list(x: paddle.Tensor, /) -> list:
    raise IvyNotImplementedException()


def gather(
    params: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    batch_dims: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def gather_nd(
    params: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    batch_dims: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def get_num_dims(
    x: paddle.Tensor, /, *, as_array: bool = False
) -> Union[paddle.Tensor, int]:
    raise IvyNotImplementedException()


def inplace_arrays_supported():
    raise IvyNotImplementedException()


def inplace_decrement(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
) -> ivy.Array:
    raise IvyNotImplementedException()


def inplace_increment(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
) -> ivy.Array:
    raise IvyNotImplementedException()


def inplace_update(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    raise IvyNotImplementedException()


def inplace_variables_supported():
    raise IvyNotImplementedException()


def multiprocessing(context=None):
    raise IvyNotImplementedException()


def scatter_flat(
    indices: paddle.Tensor,
    updates: paddle.Tensor,
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def scatter_nd(
    indices: paddle.Tensor,
    updates: paddle.Tensor,
    /,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def shape(x: paddle.Tensor, /, *, as_array: bool = False) -> Union[ivy.Shape, ivy.Array]:
    raise IvyNotImplementedException()


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: Optional[int] = 0,
) -> Callable:
    raise IvyNotImplementedException()
