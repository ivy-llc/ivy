# global
import numpy as np
import torch
from torch import Tensor
from typing import Union, List, Optional, Sequence

# local
import ivy
from ivy import (
    as_native_dtype,
    default_dtype,
    as_native_dev,
    default_device,
)
from ivy.functional.backends.torch.device import dev
from ivy.functional.backends.numpy.data_type import as_ivy_dtype

# noinspection PyProtectedMember
from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible


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


# noinspection PyUnboundLocalVariable,PyShadowingNames
def arange(
    start,
    stop=None,
    step=1,
    *,
    dtype: torch.dtype = None,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
):
    if stop is None:
        stop = start
        start = 0
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start

    device = as_native_dev(default_device(device))

    if dtype is None:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return torch.arange(
                start, stop, step=step, dtype=torch.int64, device=device, out=out
            ).to(torch.int32)
        else:
            return torch.arange(start, stop, step=step, device=device, out=out)
    else:
        dtype = as_native_dtype(default_dtype(dtype))
        return torch.arange(start, stop, step=step, dtype=dtype, device=device, out=out)


arange.support_native_out = True


def asarray(
    object_in,
    *,
    copy: Optional[bool] = None,
    dtype: torch.dtype = None,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
):
    if isinstance(object_in, torch.Tensor) and dtype is None:
        dtype = object_in.dtype
    elif (
        isinstance(object_in, (list, tuple, dict))
        and len(object_in) != 0
        and dtype is None
    ):
        dtype = default_dtype(item=object_in, as_native=True)
        if copy is True:
            return torch.as_tensor(object_in, dtype=dtype).clone().detach().to(device)
        else:
            return torch.as_tensor(object_in, dtype=dtype).to(device)

    elif isinstance(object_in, np.ndarray) and dtype is None:
        dtype = as_native_dtype(as_ivy_dtype(object_in.dtype))
    else:
        dtype = as_native_dtype((default_dtype(dtype, object_in)))

    if copy is True:
        return torch.as_tensor(object_in, dtype=dtype).clone().detach().to(device)
    else:
        return torch.as_tensor(object_in, dtype=dtype).to(device)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    return torch.empty(
        shape,
        dtype=as_native_dtype(default_dtype(dtype)),
        device=as_native_dev(default_device(device)),
        out=out,
    )


empty.support_native_out = True


def empty_like(
    x: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if device is None:
        device = dev(x)
    dtype = as_native_dtype(dtype)
    return torch.empty_like(x, dtype=dtype, device=as_native_dev(device))


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = as_native_dtype(default_dtype(dtype))
    device = as_native_dev(default_device(device))
    if n_cols is None:
        n_cols = n_rows
    if batch_shape is None:
        return torch.eye(n_rows, n_cols, dtype=dtype, device=device, out=out)
    i = torch.eye(n_rows, n_cols, dtype=dtype, device=device)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]
    return_mat = torch.reshape(i, reshape_dims).repeat(tile_dims)

    # k=index of the diagonal. A positive value refers to an upper diagonal,
    # a negative value to a lower diagonal, and 0 to the main diagonal.
    # Default: 0.
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
                torch.zeros([n_rows, k], dtype=dtype, device=device, out=out),
                i[:, : n_cols - k],
            ],
            1,
        )
        ret = torch.reshape(mat, reshape_dims).repeat(tile_dims)
    else:
        ret = torch.zeros(
            batch_shape + [n_rows, n_cols], dtype=dtype, device=device, out=out
        )
    if out is not None:
        return ivy.inplace_update(out, ret)
    return ret


eye.support_native_out = True


def from_dlpack(x, *, out: Optional[torch.Tensor] = None):
    x = x.detach() if x.requires_grad else x
    return torch.utils.dlpack.from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, torch.dtype]] = None,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    dtype = ivy.default_dtype(dtype, item=fill_value, as_native=True)
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    if isinstance(shape, int):
        shape = (shape,)
    return torch.full(
        shape,
        fill_value,
        dtype=dtype,
        device=device,
        out=out,
    )


full.support_native_out = True


def full_like(
    x: torch.Tensor,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Union[ivy.Dtype, torch.dtype]] = None,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.default_dtype(dtype, item=fill_value, as_native=True)
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    if device is None:
        device = dev(x)
    dtype = as_native_dtype(dtype)
    return torch.full_like(x, fill_value, dtype=dtype, device=default_device(device))


def linspace(
    start,
    stop,
    num,
    axis=None,
    endpoint=True,
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
):
    if not endpoint:
        ans = linspace_helper(start, stop, num + 1, axis, device=device, dtype=dtype)[
            :-1
        ]
    else:
        ans = linspace_helper(start, stop, num, axis, device=device, dtype=dtype)
    if dtype is None:
        dtype = torch.float32
    ans = ans.type(dtype)
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
    return ans


linspace.support_native_out = True


def linspace_helper(start, stop, num, axis=None, device=None, dtype=None):
    num = num.detach().numpy().item() if isinstance(num, torch.Tensor) else num
    start_is_array = isinstance(start, torch.Tensor)
    stop_is_array = isinstance(stop, torch.Tensor)
    linspace_method = torch.linspace
    device = default_device(device)
    sos_shape = []
    if start_is_array:
        start_shape = list(start.shape)
        sos_shape = start_shape
        if num == 1:
            if axis is not None:
                return start.unsqueeze(axis).to(as_native_dev(device))
            else:
                return start.to(as_native_dev(device))
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
                    device=as_native_dev(device),
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
                linspace_method(
                    strt, stp, num, device=as_native_dev(device), dtype=dtype
                )
                for strt, stp in zip(start, stop)
            ]
        torch.cat(res, -1).reshape(start_shape + [num])
    elif start_is_array and not stop_is_array:
        if num < start.shape[0]:
            start = start.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(
                torch.ones_like(start, device=as_native_dev(device), dtype=dtype) * stop
            )
        else:
            res = [
                linspace_method(
                    strt, stop, num, device=as_native_dev(device), dtype=dtype
                )
                for strt in start
            ]
    elif not start_is_array and stop_is_array:
        if num < stop.shape[0]:
            stop = stop.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [
                torch.ones_like(stop, device=as_native_dev(device), dtype=dtype) * start
            ]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [
                linspace_method(
                    start, stp, num, device=as_native_dev(device), dtype=dtype
                )
                for stp in stop
            ]
    else:
        return linspace_method(
            start, stop, num, device=as_native_dev(device), dtype=dtype
        )
    res = torch.cat(res, -1).reshape(sos_shape + [num])
    if axis is not None:
        res = torch.transpose(res, axis, -1)
    return res.to(as_native_dev(device))


def meshgrid(*arrays: torch.Tensor, indexing="xy") -> List[torch.Tensor]:
    return list(torch.meshgrid(*arrays, indexing=indexing))


# noinspection PyShadowingNames
def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype_val: torch.dtype = as_native_dtype(dtype)
    device = default_device(device)
    return torch.ones(shape, dtype=dtype_val, device=as_native_dev(device))


ones.support_native_out = True


def ones_like(
    x: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if device is None:
        device = dev(x)
    dtype = as_native_dtype(dtype)
    return torch.ones_like(x, dtype=dtype, device=as_native_dev(device))


def tril(
    x: torch.Tensor, k: int = 0, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.tril(x, diagonal=k, out=out)


tril.support_native_out = True


def triu(
    x: torch.Tensor, k: int = 0, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.triu(x, diagonal=k, out=out)


triu.support_native_out = True


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device, out=out)


zeros.support_native_out = True


def zeros_like(
    x: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if device is None:
        device = dev(x)
    if dtype is not None:
        if isinstance(x, bool):
            x = int(x)
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        return torch.zeros_like(x, dtype=dtype, device=as_native_dev(device))
    return torch.zeros_like(x, device=as_native_dev(device))


# Extra #
# ------#


array = asarray


def logspace(
    start,
    stop,
    num,
    base=10.0,
    axis=None,
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
):
    power_seq = linspace(
        start, stop, num, axis, dtype=None, device=default_device(device)
    )
    return base**power_seq


logspace.support_native_out = True
