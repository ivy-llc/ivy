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
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version
import multiprocessing as _multiprocessing
from .elementwise import _elementwise_helper


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, paddle.Tensor):
        if exclusive and not x.stop_gradient:
            return False
        return True
    return False


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def array_equal(x0: paddle.Tensor, x1: paddle.Tensor, /) -> bool:
    return bool(ivy.all(ivy.equal(x0, x1)))


def container_types():
    return []


def current_backend_str() -> str:
    return "paddle"


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def get_item(x: paddle.Tensor, query: Union[paddle.Tensor, Tuple]) -> paddle.Tensor:
    # regular queries x[idx_1,idx_2,...,idx_i]
    if isinstance(query, tuple):
        if x.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
            return x.cast("float32").__getitem__(query).cast(x.dtype)
        return x.__getitem__(query)

    if not ivy.is_native_array(query):
        query = paddle.to_tensor(query, dtype="int64")

    # masked queries x[bool_1,bool_2,...,bool_i]
    if query.dtype == paddle.bool:
        if x.dtype in [
            paddle.int8,
            paddle.int16,
            paddle.uint8,
            paddle.float16,
            paddle.complex64,
            paddle.complex128,
            paddle.bool,
        ]:
            if paddle.is_complex(x):
                return paddle.complex(
                    paddle.masked_select(x.real(), query),
                    paddle.masked_select(x.imag(), query),
                )
            return paddle.masked_select(x.cast("float32"), query).cast(x.dtype)
        return paddle.masked_select(x, query)

    query = query.cast("int64")
    # array queries idx = Tensor(idx_1,idx_2,...,idx_i), x[idx]
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.complex(
                x.real().__getitem__(query), x.imag().__getitem__(query)
            )
        return x.cast("float32").__getitem__(query).cast(x.dtype)
    return x.__getitem__(query)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def to_scalar(x: paddle.Tensor, /) -> Number:
    if isinstance(x, (Number, complex)):
        return x
    return x.item()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def to_list(x: paddle.Tensor, /) -> list:
    return x.tolist()


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "uint16",
                "bfloat16",
                "int8",
                "int16",
                "int32",
                "int64",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def gather(
    params: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    batch_dims: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axis = axis % paddle.Tensor.ndimension(params)
    batch_dims = batch_dims % paddle.Tensor.ndimension(params)
    ivy.utils.assertions.check_gather_input_valid(params, indices, axis, batch_dims)
    if batch_dims == 0:
        result = paddle.gather(params, paddle.reshape(indices, shape=[-1]), axis=axis)
    else:
        params_list = [p for p in params]
        indices_list = [i for i in indices]
        for b in range(1, batch_dims):
            params_list = [p1 for p in params_list for p1 in p]
            indices_list = [i1 for i in indices_list for i1 in i]
        result = []
        for p, i in zip(params_list, indices_list):
            result.append(
                paddle.gather(p, paddle.reshape(i, shape=[-1]), axis=axis - batch_dims)
            )
        result = paddle.concat(result, axis=0)
    new_shape = (
        params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis + 1 :]
    )
    return paddle.reshape(result, shape=new_shape)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def gather_nd(
    params: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    batch_dims: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def get_num_dims(
    x: paddle.Tensor, /, *, as_array: bool = False
) -> Union[paddle.Tensor, int]:
    return paddle.to_tensor(x.ndim) if as_array else x.ndim


def inplace_arrays_supported():
    # there are some operations that support inplace updates
    # but it's not supported in all functions
    return False


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_decrement(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_increment(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_update(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
) -> ivy.Array:
    if ivy.is_array(x) and ivy.is_array(val):
        (x_native, val_native), _ = ivy.args_to_native(x, val)

        if val_native.shape == x_native.shape:
            if x_native.dtype != val_native.dtype:
                x_native = x_native.astype(val_native.dtype)
            paddle.assign(val_native, x_native)
        else:
            x_native = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
        return x
    else:
        return val


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_variables_supported():
    return False


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def shape(
    x: paddle.Tensor, /, *, as_array: bool = False
) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(x.shape, dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: Optional[int] = 0,
) -> Callable:
    raise IvyNotImplementedException()


def isin(
    elements: paddle.Tensor,
    test_elements: paddle.Tensor,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
