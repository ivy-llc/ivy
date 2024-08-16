# global
import copy
from numbers import Number
from typing import Union, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_unsupported_device_and_dtypes,
)
from ivy.functional.ivy.creation import (
    _asarray_to_native_arrays_and_back,
    _asarray_infer_device,
    _asarray_infer_dtype,
    _asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
    _asarray_inputs_to_native_shapes,
    _remove_np_bfloat16,
)
from . import backend_version


# noinspection PyProtectedMember


# Array API Standard #
# -------------------#


def _differentiable_linspace(start, stop, num, *, device, dtype=None):
    if num == 1:
        return torch.unsqueeze(start, 0)
    n_m_1 = num - 1
    increment = (stop - start) / n_m_1
    increment_tiled = increment.repeat(n_m_1)
    increments = increment_tiled * torch.linspace(
        1, n_m_1, n_m_1, device=device, dtype=dtype
    )
    res = torch.cat(
        (torch.unsqueeze(torch.tensor(start, dtype=dtype), 0), start + increments), 0
    )
    return res


@with_unsupported_dtypes({"2.2 and below": ("complex",)}, backend_version)
def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, torch.dtype]] = None,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
            return torch.arange(start, stop, step, dtype=torch.int64, device=device).to(
                torch.int32
            )
        else:
            return torch.arange(start, stop, step, device=device)
    else:
        dtype = ivy.as_native_dtype(ivy.default_dtype(dtype=dtype))
        return torch.arange(start, stop, step, dtype=dtype, device=device)


arange.support_native_out = True


def _stack_tensors(x, dtype):
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x[i] = _stack_tensors(item, dtype)
        x = torch.stack(x)
    else:
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], torch.Tensor):
                x = torch.stack([torch.as_tensor(i, dtype=dtype) for i in x])
            else:
                x = torch.as_tensor(x, dtype=dtype)
    return x


@with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, backend_version)
@_asarray_to_native_arrays_and_back
@_asarray_infer_device
@_asarray_handle_nestable
@_asarray_inputs_to_native_shapes
@_asarray_infer_dtype
def asarray(
    obj: Union[
        torch.Tensor,
        np.ndarray,
        torch.Size,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, torch.dtype]] = None,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    obj = ivy.nested_map(_remove_np_bfloat16, obj, shallow=False)
    if isinstance(obj, Sequence) and len(obj) != 0:
        contain_tensor = ivy.nested_any(obj, lambda x: isinstance(x, torch.Tensor))
        # if `obj` is a list of specifically tensors or
        # a multidimensional list which contains a tensor
        if contain_tensor:
            ret = _stack_tensors(obj, dtype).to(device)
            return ret.clone().detach() if copy else ret
    try:
        ret = torch.as_tensor(obj, dtype=dtype, device=device)
    except ValueError as e:
        if "At least one stride in the given numpy array is negative" in str(e):
            ret = torch.as_tensor(obj.copy(), dtype=dtype, device=device)
        else:
            raise
    return ret.clone().detach() if copy else ret


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty(
        shape,
        dtype=dtype,
        device=device,
        out=out,
    )


empty.support_native_out = True


def empty_like(
    x: torch.Tensor,
    /,
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=dtype, device=device)


@with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, backend_version)
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if n_cols is None:
        n_cols = n_rows
    if batch_shape is None:
        batch_shape = []
    i = torch.eye(n_rows, n_cols, dtype=dtype, device=device)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]
    return_mat = torch.reshape(i, reshape_dims).repeat(tile_dims)

    # k=index of the diagonal. A positive value refers to an upper diagonal,
    # a negative value to a lower diagonal, and 0 to the main diagonal.
    # Default: ``0``.
    # value of k ranges from -n_rows < k < n_cols

    if k == 0:  # refers to the main diagonal
        ret = return_mat

    # when k is negative
    elif -n_rows < k < 0:
        mat = torch.concat(
            [
                torch.zeros([-k, n_cols], dtype=dtype, device=device, out=out),
                i[: n_rows + k],
            ],
            0,
        )
        ret = torch.reshape(mat, reshape_dims).repeat(tile_dims)

    # when k is positive
    elif 0 < k < n_cols:
        mat = torch.concat(
            [
                torch.zeros([n_rows, k], dtype=dtype, device=device),
                i[:, : n_cols - k],
            ],
            1,
        )
        ret = torch.reshape(mat, reshape_dims).repeat(tile_dims)
        if out is not None:
            return ivy.inplace_update(out, ret)
    else:
        ret = torch.zeros(
            batch_shape + [n_rows, n_cols], dtype=dtype, device=device, out=out
        )
    return ret


eye.support_native_out = True


def to_dlpack(x, /, *, out: Optional[torch.Tensor] = None):
    return torch.to_dlpack(x)


def from_dlpack(x, /, *, out: Optional[torch.Tensor] = None):
    return torch.from_dlpack(x)


@with_unsupported_dtypes({"2.2.0 and below": ("bfloat16",)}, backend_version)
def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, torch.dtype]] = None,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    if isinstance(shape, int):
        shape = (shape,)

    shape = tuple(int(dim) for dim in shape)
    fill_value = torch.tensor(fill_value, dtype=dtype)

    return torch.full(
        size=shape,
        fill_value=fill_value,
        dtype=dtype,
        device=device,
        out=out,
    )


full.support_native_out = True


def full_like(
    x: torch.Tensor,
    /,
    fill_value: Number,
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.full_like(x, fill_value, dtype=dtype, device=device)


def _slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)


@with_unsupported_device_and_dtypes(
    {"2.2 and below": {"cpu": ("float16",)}}, backend_version
)
def linspace(
    start: Union[torch.Tensor, float],
    stop: Union[torch.Tensor, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = -1
    if not endpoint:
        if dtype is not None:
            ans = linspace_helper(
                start, stop, num + 1, axis, dtype=dtype, device=device
            )
        else:
            ans = linspace_helper(start, stop, num + 1, axis, device=device)
        if axis < 0:
            axis += len(ans.shape)
        ans = ans[_slice_at_axis(slice(None, -1), axis)]
    else:
        if dtype is not None:
            ans = linspace_helper(start, stop, num, axis, dtype=dtype, device=device)
        else:
            ans = linspace_helper(start, stop, num, axis, device=device)
    if (
        endpoint
        and ans.shape[0] > 1
        and (not isinstance(start, torch.Tensor))
        and (not isinstance(stop, torch.Tensor))
    ):
        ans[-1] = stop
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, torch.Tensor))
        and (not isinstance(stop, torch.Tensor))
        and ans[0] != start
    ):
        ans[0] = start
    if "int" in str(dtype) and torch.is_floating_point(ans):
        ans = torch.floor(ans)
    return ans.to(dtype)


linspace.support_native_out = True


def linspace_helper(start, stop, num, axis=None, *, dtype=None, device):
    num = num.detach().numpy().item() if isinstance(num, torch.Tensor) else num
    start_is_array = isinstance(start, torch.Tensor)
    stop_is_array = isinstance(stop, torch.Tensor)
    linspace_method = torch.linspace
    sos_shape = []
    if start_is_array:
        start_shape = list(start.shape)
        sos_shape = start_shape
        if num == 1:
            if axis is not None:
                return start.unsqueeze(axis).to(device)
            else:
                return start.unsqueeze(-1).to(device)
        start = start.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if start.requires_grad else torch.linspace
        )
    if stop_is_array:
        stop_shape = list(stop.shape)
        sos_shape = stop_shape
        if num == 1:
            return (
                torch.ones(
                    stop_shape[:axis] + [1] + stop_shape[axis:],
                    device=device,
                )
                * start
            )
        stop = stop.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if stop.requires_grad else torch.linspace
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
                linspace_method(strt, stp, num, device=device)
                for strt, stp in zip(start, stop)
            ]
    elif start_is_array and not stop_is_array:
        if num < start.shape[0]:
            start = start.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(torch.ones_like(start, device=device) * stop)
        else:
            res = [linspace_method(strt, stop, num, device=device) for strt in start]
    elif not start_is_array and stop_is_array:
        if num < stop.shape[0]:
            stop = stop.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [torch.ones_like(stop, device=device) * start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [linspace_method(start, stp, num, device=device) for stp in stop]
    else:
        return linspace_method(start, stop, num, dtype=dtype, device=device)
    res = torch.cat(res, -1).reshape(sos_shape + [num])
    if axis is not None:
        ndim = res.ndim
        perm = list(range(0, ndim - 1))
        perm.insert(axis % (ndim + 1), ndim - 1)
        res = res.permute(perm)
    return res.to(device)


def meshgrid(
    *arrays: torch.Tensor,
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    if not sparse:
        return list(torch.meshgrid(*arrays, indexing=indexing))

    sd = (1,) * len(arrays)
    res = [
        torch.reshape(torch.as_tensor(a), (sd[:i] + (-1,) + sd[i + 1 :]))
        for i, a in enumerate(arrays)
    ]

    if indexing == "xy" and len(arrays) > 1:
        res[0] = torch.reshape(res[0], (1, -1) + sd[2:])
        res[1] = torch.reshape(res[1], (-1, 1) + sd[2:])

    return res


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ones(shape, dtype=dtype, device=device, out=out)


ones.support_native_out = True


def ones_like_v_0p4p0_and_above(
    x: torch.Tensor,
    /,
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ones_like(x, dtype=dtype, device=device)


def ones_like_v_0p3p0_to_0p3p1(
    x: torch.Tensor,
    /,
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ones_like(x, out=out)


def ones_like_v_0p1p12_to_0p2p0(
    x: torch.Tensor,
    /,
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
):
    if len(x.shape) == 1:
        for i in range(x.shape[0]):
            x[i] = 1
        return x
    for i in range(x.shape[0]):
        x[i, :] = ones_like_v_0p1p12_to_0p2p0(x[i, :], dtype=dtype)
    return x


def tril(
    x: torch.Tensor, /, *, k: int = 0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.tril(x, diagonal=k, out=out)


tril.support_native_out = True


def triu(
    x: torch.Tensor, /, *, k: int = 0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.triu(x, diagonal=k, out=out)


triu.support_native_out = True


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device, out=out)


zeros.support_native_out = True


def zeros_like(
    x: torch.Tensor,
    /,
    *,
    dtype: torch.dtype,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.zeros_like(x, dtype=dtype, device=device)


# Extra #
# ------#


array = asarray


def copy_array(
    x: torch.Tensor,
    *,
    to_ivy_array: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if to_ivy_array:
        return ivy.to_ivy(x.clone())
    return x.clone()


def one_hot(
    indices: torch.Tensor,
    depth: int,
    /,
    *,
    on_value: Optional[torch.Tensor] = None,
    off_value: Optional[torch.Tensor] = None,
    axis: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: torch.device = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    on_none = on_value is None
    off_none = off_value is None

    if dtype is None:
        if on_none and off_none:
            dtype = torch.float32
        else:
            if not on_none:
                dtype = torch.tensor(on_value).dtype
            elif not off_none:
                dtype = torch.tensor(off_value).dtype
    else:
        dtype = ivy.as_native_dtype(dtype)

    on_value = torch.tensor(1.0) if on_none else torch.tensor(on_value, dtype=dtype)
    off_value = torch.tensor(0.0) if off_none else torch.tensor(off_value, dtype=dtype)

    res = torch.nn.functional.one_hot(indices.to(torch.int64), depth)

    if not on_none or not off_none:
        res = torch.where(res == 1, on_value, off_value)

    if axis is not None:
        res = torch.moveaxis(res, -1, axis)

    return res.to(device, dtype)


def frombuffer(
    buffer: bytes,
    dtype: Optional[torch.dtype] = float,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> torch.Tensor:
    buffer_copy = copy.deepcopy(buffer)
    dtype = ivy.as_native_dtype(dtype)

    return torch.frombuffer(buffer_copy, dtype=dtype, count=count, offset=offset)


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: torch.device = None,
) -> Tuple[torch.Tensor]:
    n_cols = n_rows if n_cols is None else n_cols
    return tuple(
        torch.triu_indices(
            row=n_rows, col=n_cols, offset=k, dtype=torch.int64, device=device
        )
    )
