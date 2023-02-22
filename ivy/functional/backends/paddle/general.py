"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""
# global
from functools import reduce
from numbers import Number
from operator import mul
from typing import Optional, Union, Sequence, Callable, List, Tuple
import paddle
import numpy as np

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, paddle.Tensor):
        if exclusive and not x.stop_gradient:
            return False
        return True
    return False


def array_equal(x0: paddle.Tensor, x1: paddle.Tensor, /) -> bool:
    raise IvyNotImplementedException()


def container_types():
    return []


def current_backend_str() -> str:
    return "paddle"


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "int8", "int16", "uint8",)}, backend_version)
def get_item(
    x: paddle.Tensor,
    query: Union[paddle.Tensor, Tuple]
) -> paddle.Tensor:
    if isinstance(query, tuple):
        return x.__getitem__(query)

    if not ivy.is_native_array(query):
        query = paddle.to_tensor(query)

    dtype = ivy.dtype(query, as_native=True)
    x_dtype = ivy.dtype(x, as_native=True)

    if dtype is paddle.bool:
        return paddle.masked_select(x, query)

    if x_dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        ret = paddle.cast(x, 'float32').__getitem__(tuple(query))
        return paddle.cast(ret, x_dtype)

    return x.__getitem__(tuple(query))


def to_numpy(
    x: Union[paddle.Tensor, List[paddle.Tensor]], /, *, copy: bool = True
) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(x, (float, int, bool)):
        return x
    elif isinstance(x, np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    elif paddle.is_tensor(x):
        if copy:
            return np.array(x)
        else:
            return np.asarray(x)
    elif isinstance(x, list):
        return [ivy.to_numpy(u) for u in x]
    raise ivy.utils.exceptions.IvyException("Expected a Paddle Tensor.")


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


def shape(
    x: paddle.Tensor, /, *, as_array: bool = False
) -> Union[ivy.Shape, ivy.Array]:
    raise IvyNotImplementedException()


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: Optional[int] = 0,
) -> Callable:
    raise IvyNotImplementedException()
