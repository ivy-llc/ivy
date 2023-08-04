# global
import struct
from numbers import Number
from typing import Union, List, Optional, Sequence, Tuple

import numpy as np
import paddle
import ivy.functional.backends.paddle as paddle_backend

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
)
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    asarray_infer_dtype,
    NestedSequence,
    SupportsBufferProtocol,
    asarray_inputs_to_native_shapes,
    _remove_np_bfloat16,
)
from . import backend_version
from paddle.device import core
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
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if stop is None:
        stop = start
        start = 0
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start
    if dtype is None:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return to_device(
                paddle.arange(start, stop, step, dtype=paddle.int32), device
            )

        elif (
            isinstance(start, float)
            or isinstance(stop, float)
            or isinstance(step, float)
        ):
            return to_device(
                paddle.arange(start, stop, step, dtype=paddle.float32), device
            )

        else:
            return to_device(paddle.arange(start, stop, step), device)
    else:
        return to_device(paddle.arange(start, stop, step).cast(dtype), device)


@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
@asarray_inputs_to_native_shapes
@asarray_infer_dtype
def asarray(
    obj: Union[
        paddle.Tensor,
        np.ndarray,
        bool,
        int,
        float,
        list,
        NestedSequence,
        SupportsBufferProtocol,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    device = ivy.as_native_dev(device)
    if isinstance(obj, paddle.Tensor):
        if copy:
            ret = obj.clone().detach()
            ret.stop_gradient = obj.stop_gradient
        else:
            ret = obj
        return to_device(ret, device).astype(dtype)

    elif isinstance(obj, (Number, bool, complex)):
        return paddle_backend.squeeze(
            paddle.to_tensor(obj, dtype=dtype, place=device), axis=0
        )
    obj = ivy.nested_map(obj, _remove_np_bfloat16, shallow=False)
    return paddle.to_tensor(obj, dtype=dtype, place=device)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(shape, int):
        shape = [shape]
    return to_device(paddle.empty(shape=shape).cast(dtype), device)


def empty_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.empty(shape=x.shape).cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "uint8",
                "int8",
                "int16",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if n_cols is None:
        n_cols = n_rows
    if batch_shape is None:
        batch_shape = []
    i = to_device(paddle.eye(n_rows, n_cols, dtype=dtype), device)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]

    # handle index of the diagonal k
    if k == 0:
        return paddle.reshape(i, reshape_dims)

    elif -n_rows < k < 0:
        mat = paddle.concat(
            [
                to_device(paddle.zeros([-k, n_cols], dtype=dtype), device),
                i[: n_rows + k],
            ],
            0,
        )
        return paddle.tile(paddle.reshape(mat, reshape_dims), tile_dims)

    elif 0 < k < n_cols:
        mat = paddle.concat(
            [
                to_device(paddle.zeros([n_rows, k], dtype=dtype), device),
                i[:, : n_cols - k],
            ],
            1,
        )
        return paddle.tile(paddle.reshape(mat, reshape_dims), tile_dims)
    else:
        return to_device(
            paddle.zeros(batch_shape + [n_rows, n_cols], dtype=dtype), device
        )


def from_dlpack(x, /, *, out: Optional[paddle.Tensor] = None):
    x_d = paddle.utils.dlpack.to_dlpack(x)
    return paddle.utils.dlpack.from_dlpack(x_d)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if dtype is None:
        dtype = ivy.default_dtype(item=fill_value)
    if not isinstance(shape, Sequence):
        shape = [shape]
    if ivy.as_native_dtype(dtype) is paddle.int8:
        return paddle.full(shape=shape, fill_value=fill_value).cast(dtype)
    else:
        return paddle.full(shape=shape, fill_value=fill_value, dtype=dtype)


def full_like(
    x: paddle.Tensor,
    /,
    fill_value: Number,
    *,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle_backend.full(
        shape=x.shape, fill_value=fill_value, dtype=dtype, device=device
    )


def _linspace_helper(start, stop, num, axis=None, *, dtype=None):
    num = num.detach().item() if isinstance(num, paddle.Tensor) else num
    start_is_array = isinstance(start, paddle.Tensor)
    stop_is_array = isinstance(stop, paddle.Tensor)
    linspace_method = paddle.linspace
    sos_shape = []
    if start_is_array:
        start_shape = start.shape
        sos_shape = start_shape
        if num == 1:
            if axis is not None:
                return paddle_backend.expand_dims(start, axis=axis)
            else:
                return paddle_backend.expand_dims(start, axis=-1)
        start = start.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if not start.stop_gradient else paddle.linspace
        )
    if stop_is_array:
        stop_shape = stop.shape
        sos_shape = stop_shape
        if num == 1:
            return (
                paddle_backend.ones(stop_shape[:axis] + [1] + stop_shape[axis:]) * start
            )
        stop = stop.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if not stop.stop_gradient else paddle.linspace
        )
    if start_is_array and stop_is_array:
        if num < start.shape[0]:
            start = paddle_backend.expand_dims(start, axis=-1)
            stop = paddle_backend.expand_dims(stop, axis=-1)
            diff = paddle_backend.subtract(stop, start)
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [
                linspace_method(strt, stp, num)
                for strt, stp in zip(
                    paddle_backend.unstack(start, keepdims=True),
                    paddle_backend.unstack(stop, keepdims=True),
                )
            ]
    elif start_is_array and not stop_is_array:
        if num < start.shape[0]:
            start = paddle_backend.expand_dims(start, axis=axis)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(paddle.ones(start.shape).astype(start.dtype) * stop)
        else:
            res = [linspace_method(strt, stop, num) for strt in start]
    elif not start_is_array and stop_is_array:
        if num < stop.shape[0]:
            stop = paddle_backend.expand_dims(stop, axis=-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [paddle.ones(stop.shape).astype(stop.dtype) * start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [linspace_method(start, stp, num) for stp in stop]
    else:
        return linspace_method(start, stop, num, dtype=dtype)
    res = paddle_backend.concat(res, axis=-1).reshape(sos_shape + [num])
    if axis is not None:
        ndim = res.ndim
        perm = list(range(ndim - 1))
        perm.insert(axis % (ndim + 1), ndim - 1)
        res = paddle_backend.permute_dims(res, perm)
    return res


def _differentiable_linspace(start, stop, num, *, dtype=None):
    start = ivy.to_native(start)
    num = paddle.to_tensor(num, stop_gradient=False)
    if num == 1:
        return paddle_backend.expand_dims(start, axis=0)
    n_m_1 = paddle_backend.subtract(num, 1)
    increment = paddle_backend.divide(paddle_backend.subtract(stop, start), n_m_1)
    increment_tiled = paddle_backend.repeat(increment, n_m_1)
    increments = paddle_backend.multiply(
        increment_tiled,
        paddle.linspace(1, n_m_1, n_m_1.cast(paddle.int32), dtype=dtype),
    )
    if isinstance(start, int) or start.ndim == 0:
        start = paddle_backend.expand_dims(start, axis=0)
    res = paddle_backend.concat((start, paddle_backend.add(start, increments)), axis=0)
    return res.cast(dtype)


def _slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("uint16", "bfloat16", "float16")}}, backend_version
)
def linspace(
    start: Union[paddle.Tensor, float],
    stop: Union[paddle.Tensor, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if not isinstance(start, (paddle.Tensor, int)):
        start = paddle.to_tensor(start)

    if not isinstance(start, (paddle.Tensor, int)):
        start = paddle.to_tensor(stop)

    if axis is None:
        axis = -1
    if not endpoint:
        if dtype is not None:
            ans = _linspace_helper(start, stop, num + 1, axis, dtype=dtype)
        else:
            ans = _linspace_helper(start, stop, num + 1, axis)
        if axis < 0:
            axis += len(ans.shape)
        ans = paddle_backend.get_item(ans, _slice_at_axis(slice(None, -1), axis))
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
    if ivy.is_ivy_array(ans):
        ans = paddle.to_tensor(ans.data)
    if "int" in str(dtype) and paddle.is_floating_point(ans):
        ans = paddle.floor(ans)
    return to_device(ans.cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "float16",
                "complex",
                "bool",
            )
        }
    },
    backend_version,
)
def meshgrid(
    *arrays: paddle.Tensor,
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[paddle.Tensor] = None,
) -> List[paddle.Tensor]:
    if len(arrays) == 1:
        return arrays
    if not sparse:
        if indexing == "ij":
            return paddle.meshgrid(*arrays)
        elif indexing == "xy":
            index_switch = lambda x: (
                paddle_backend.swapaxes(x, 0, 1) if x.ndim > 1 else x
            )
            arrays = list(map(index_switch, arrays))
            ret = paddle.meshgrid(*arrays)
            return list(map(index_switch, ret))
        else:
            raise ValueError(f"indexing must be either 'ij' or 'xy', got {indexing}")

    sd = (1,) * len(arrays)
    res = [
        paddle.reshape(paddle.to_tensor(a), (sd[:i] + (-1,) + sd[i + 1 :]))
        for i, a in enumerate(arrays)
    ]
    if indexing == "xy" and len(arrays) > 1:
        res[0] = paddle.reshape(res[0], (1, -1) + sd[2:])
        res[1] = paddle.reshape(res[1], (-1, 1) + sd[2:])

    return res


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.ones(shape=shape).cast(dtype), device)


def ones_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle_backend.ones(shape=x.shape, dtype=dtype, device=device)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "complex",
            )
        }
    },
    backend_version,
)
def tril(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.tril(x=x, diagonal=k)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "complex",
            )
        }
    },
    backend_version,
)
def triu(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.triu(x=x, diagonal=k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.zeros(shape=shape).cast(dtype), device)


def zeros_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle_backend.zeros(shape=x.shape, dtype=dtype, device=device)


# Extra #
# ------#


array = asarray


def copy_array(
    x: paddle.Tensor,
    *,
    to_ivy_array: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if 0 in x.shape:
        new_arr = paddle.empty(x.shape, dtype=x.dtype)
    else:
        new_arr = x.clone()
    if to_ivy_array:
        return ivy.to_ivy(new_arr)
    return new_arr


def one_hot(
    indices: paddle.Tensor,
    depth: int,
    /,
    *,
    on_value: Optional[paddle.Tensor] = None,
    off_value: Optional[paddle.Tensor] = None,
    axis: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: core.Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    on_none = on_value is None
    off_none = off_value is None
    expand_ret = False
    if indices.ndim == 0:
        expand_ret = True
        indices = indices.cast("int64").unsqueeze(0)
    if dtype is None:
        if on_none and off_none:
            dtype = paddle.float32
        else:
            if not on_none:
                dtype = paddle.to_tensor(on_value).dtype
            elif not off_none:
                dtype = paddle.to_tensor(off_value).dtype
    else:
        dtype = ivy.as_native_dtype(dtype)

    on_value = (
        paddle.to_tensor(1.0, dtype="float32")
        if on_none
        else paddle.to_tensor(on_value, dtype="float32")
    )
    off_value = (
        paddle.to_tensor(0.0, dtype="float32")
        if off_none
        else paddle.to_tensor(off_value, dtype="float32")
    )

    res = paddle.nn.functional.one_hot(indices.cast(paddle.int64), depth)

    if not on_none or not off_none:
        res = paddle.where(res == 1, on_value, off_value)

    if axis is not None:
        res = paddle.moveaxis(res, -1, axis)
    if expand_ret:
        res = res.squeeze()
    return to_device(res.cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("complex64", "complex128")}},
    backend_version,
)
def frombuffer(
    buffer: bytes,
    dtype: Optional[paddle.dtype] = float,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> paddle.Tensor:
    dtype_bytes = int(ivy.Dtype(dtype).dtype_bits / 8)
    if str(dtype) == "bool":
        dtype_bytes = 1
    dtype_str = str(dtype)
    struct_format = {
        "bool": "?",
        "int8": "b",
        "int16": "h",
        "int32": "i",
        "int64": "q",
        "uint8": "B",
        "float16": "e",
        "float32": "f",
        "float64": "d",
    }
    ret = []
    for i in range(0, len(buffer), dtype_bytes):
        x = struct.unpack(struct_format[dtype_str], buffer[i : i + dtype_bytes])
        ret = ret + list(x)
    if offset > 0:
        offset = int(offset / dtype_bytes)
    if count > -1:
        ret = ret[offset : offset + count]
    else:
        ret = ret[offset:]
    ret = paddle.to_tensor(ret, dtype=dtype)

    return ret


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: core.Place,
) -> Tuple[paddle.Tensor]:
    # special case due to inconsistent behavior when n_cols=1 and n_rows=0
    if n_cols == 1 and n_rows == 0:
        return paddle.to_tensor([], place=device, dtype="int64"), paddle.to_tensor(
            [], place=device, dtype="int64"
        )
    return tuple(
        to_device(
            paddle.triu_indices(n_rows, col=n_cols, offset=k, dtype="int64"), device
        )
    )
