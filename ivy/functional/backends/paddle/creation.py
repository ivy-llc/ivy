# global

from numbers import Number
from typing import Union, List, Optional, Sequence

import numpy as np
import paddle

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_unsupported_device_and_dtypes,
    _get_first_array,

)
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
)
from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException
from paddle.fluid.libpaddle import Place
from ivy.functional.backends.paddle.device import to_device

# Array API Standard #
# -------------------#


def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def _stack_tensors(x, dtype):
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x[i] = _stack_tensors(item, dtype)
        x = paddle.stack(x)
    else:
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], paddle.Tensor):
                x = paddle.stack([paddle.to_tensor(i, dtype=dtype) for i in x])
            else:
                x = paddle.to_tensor(x, dtype=dtype)
    return x


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
def asarray(
    obj: Union[
        paddle.Tensor,
        np.ndarray,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # TODO: Implement device support

    if isinstance(obj, paddle.Tensor) and dtype is None:
        if copy is True:
            return obj.clone().detach() 
        else:
            return obj.detach()

    elif isinstance(obj, (list, tuple, dict)) and len(obj) != 0:
        contain_tensor = False
        if isinstance(obj[0], (list, tuple)):
            first_tensor = _get_first_array(obj)
            if ivy.exists(first_tensor):
                contain_tensor = True
                dtype = first_tensor.dtype
        if dtype is None:
            dtype = ivy.default_dtype(item=obj, as_native=True)

        # if `obj` is a list of specifically tensors or
        # a multidimensional list which contains a tensor
        if isinstance(obj[0], paddle.Tensor) or contain_tensor:
            if copy is True:
                return (
                    paddle.stack([paddle.to_tensor(i, dtype=dtype) for i in obj])
                    .clone()
                    .detach()

                )
            else:
                return _stack_tensors(obj, dtype)

    elif isinstance(obj, np.ndarray) and dtype is None:
        dtype = ivy.as_native_dtype(ivy.as_ivy_dtype(obj.dtype.name))

    else:
        dtype = ivy.as_native_dtype((ivy.default_dtype(dtype=dtype, item=obj)))

    if dtype == paddle.bfloat16 and isinstance(obj, np.ndarray):
        if copy is True:
            return (
                paddle.to_tensor(obj.tolist(), dtype=dtype).clone().detach()
            )
        else:
            return paddle.to_tensor(obj.tolist(), dtype=dtype)

    if copy is True:
        ret = paddle.to_tensor(obj, dtype=dtype).clone().detach()
        return ret
    else:
        ret = paddle.to_tensor(obj, dtype=dtype)
        return ret


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.empty(shape=shape, dtype=dtype), device)


def empty_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.empty_like(x=x, dtype=dtype), device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if n_cols is None:
        n_cols = n_rows
    i = paddle.eye(n_rows, n_cols, dtype=dtype)
    if batch_shape is None:
        return to_device(i, device)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]
    i = paddle.reshape(i, reshape_dims)
    return_mat = paddle.tile(i, tile_dims)
    return to_device(return_mat, device)


def from_dlpack(x, /, *, out: Optional[paddle.Tensor] = None):

    return paddle.utils.dlpack.from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.full(shape=shape, fill_value=fill_value, dtype=dtype), device)


full.support_native_out = True


def full_like(
    x: paddle.Tensor,
    /,
    fill_value: Number,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.full_like(x=x, fill_value=fill_value, dtype=dtype), device)


def _linspace_helper(start, stop, num, axis=None, *, dtype=None):
    num = num.detach().numpy().item() if isinstance(num, paddle.Tensor) else num
    start_is_array = isinstance(start, paddle.Tensor)
    stop_is_array = isinstance(stop, paddle.Tensor)
    linspace_method = paddle.linspace
    sos_shape = []
    if start_is_array:
        start_shape = start.shape
        sos_shape = start_shape
        if num == 1:
            if axis is not None:
                return start.unsqueeze(axis)
            else:
                return start.unsqueeze(-1)
        start = start.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if not start.stop_gradient else paddle.linspace
        )
    if stop_is_array:
        stop_shape = list(stop.shape)
        sos_shape = stop_shape
        if num == 1:
            return (paddle.ones(stop_shape[:axis] + [1] + stop_shape[axis:]) * start)
        stop = stop.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if not stop.stop_gradient else paddle.linspace
        )
    if start_is_array and stop_is_array:
        if num < start.shape[0]:
            start = start.unsqueeze(-1)
            stop = stop.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [
                linspace_method(strt, stp, num)
                for strt, stp in zip(start, stop)
            ]
        paddle.concat(res, -1).reshape(start_shape + [num])
    elif start_is_array and not stop_is_array:
        if num < start.shape[0]:
            start = start.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(paddle.ones_like(start) * stop)
        else:
            res = [linspace_method(strt, stop, num) for strt in start]
    elif not start_is_array and stop_is_array:
        if num < stop.shape[0]:
            stop = stop.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [paddle.ones_like(stop) * start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [linspace_method(start, stp, num) for stp in stop]
    else:
        return linspace_method(start, stop, num, dtype=dtype)
    res = paddle.concat(res, -1).reshape(sos_shape + [num])
    if axis is not None:
        ndim = res.ndim
        perm = paddle.arange(0, ndim - 1).numpy().tolist()
        perm.insert(axis % (ndim + 1), ndim - 1)
        res = paddle.transpose(res, perm)
    return res


def _differentiable_linspace(start, stop, num, *, dtype=None):
    num = paddle.to_tensor(num, stop_gradient=False)
    if num == 1:
        return paddle.unsqueeze(start, 0)
    n_m_1 = num - 1
    increment = (stop - start) / n_m_1
    increment_tiled = paddle.repeat_interleave(increment, n_m_1)
    increments = increment_tiled * paddle.linspace(
        1, n_m_1, n_m_1.cast(paddle.int32), dtype=dtype
    )
    res = paddle.concat(
        (start, start + increments), 0
    )
    return res.cast(dtype)


def _slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)


def linspace(
    start: Union[paddle.Tensor, float],
    stop: Union[paddle.Tensor, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if not isinstance(start, paddle.Tensor):
        start = paddle.to_tensor(start)

    if not isinstance(start, paddle.Tensor):
        start = paddle.to_tensor(stop)

    if not isinstance(start, paddle.Tensor):
        start = paddle.to_tensor(num)

    if axis is None:
        axis = -1
    if not endpoint:
        if dtype is not None:
            ans = _linspace_helper(
                start, stop, num + 1, axis, dtype=dtype)
        else:
            ans = _linspace_helper(start, stop, num + 1, axis)
        if axis < 0:
            axis += len(ans.shape)
        ans = ans[_slice_at_axis(slice(None, -1), axis)]
    else:
        if dtype is not None:
            ans = _linspace_helper(start, stop, num, axis, dtype=dtype)
        else:
            ans = _linspace_helper(start, stop, num, axis)
    if (
        endpoint
        and ans.shape[0] > 1
        and (not isinstance(start, paddle.Tensor))
        and (not isinstance(stop, paddle.Tensor))
    ):
        ans[-1] = stop
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, paddle.Tensor))
        and (not isinstance(stop, paddle.Tensor))
        and ans[0] != start
    ):
        ans[0] = start
    if "int" in str(dtype) and paddle.is_floating_point(ans):
        ans = paddle.floor(ans)
    return to_device(ans.cast(dtype), device)


def meshgrid(
    *arrays: paddle.Tensor,
    sparse: bool = False,
    indexing: str = "xy",
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.ones(shape=shape, dtype=dtype), device)


def ones_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.ones_like(x=x, dtype=dtype), device)


def tril(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.tril(x=x, diagonal=k)


def triu(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.triu(x=x, diagonal=k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.zeros(shape=shape, dtype=dtype), device)


def zeros_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.zeros_like(x=x, dtype=dtype), device)


# Extra #
# ------#


array = asarray


def copy_array(
    x: paddle.Tensor,
    *,
    to_ivy_array: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def one_hot(
    indices: paddle.Tensor,
    depth: int,
    /,
    *,
    on_value: Optional[paddle.Tensor] = None,
    off_value: Optional[paddle.Tensor] = None,
    axis: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
