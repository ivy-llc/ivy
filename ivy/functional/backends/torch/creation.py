# global
import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple, List, Optional, Dict
from numbers import Number

# local
from ivy import dtype_from_str, default_dtype, dev_from_str, default_device, shape_to_tuple
from ivy.functional.backends.torch.device import _callable_dev
from ivy.functional.backends.numpy.data_type import dtype_to_str as np_dtype_to_str


# Array API Standard #
# -------------------#


def asarray(object_in, dtype: Optional[str] = None, dev: Optional[str] = None, copy: Optional[bool] = None):
    dev = default_device(dev)
    if isinstance(object_in, torch.Tensor) and dtype is None:
        dtype = object_in.dtype
    elif isinstance(object_in, (list, tuple, dict)) and len(object_in) != 0 and dtype is None:
        # Temporary fix on type
        # Because default_type() didn't return correct type for normal python array
        if copy is True:
            return torch.as_tensor(object_in).clone().detach().to(dev_from_str(dev))
        else:
            return torch.as_tensor(object_in).to(dev_from_str(dev))

    elif isinstance(object_in, np.ndarray) and dtype is None:
        dtype = dtype_from_str(np_dtype_to_str(object_in.dtype))
    else:
        dtype = dtype_from_str((default_dtype(dtype, object_in)))

    if copy is True:
        return torch.as_tensor(object_in, dtype=dtype).clone().detach().to(dev_from_str(dev))
    else:
        return torch.as_tensor(object_in, dtype=dtype).to(dev_from_str(dev))


def zeros(shape: Union[int, Tuple[int]],
          dtype: Optional[torch.dtype] = None,
          device: Optional[torch.device] = None) \
        -> Tensor:
    return torch.zeros(shape, dtype=dtype_from_str(default_dtype(dtype)), device=dev_from_str(default_device(device)))


def ones(shape: Union[int, Tuple[int]],
         dtype: Optional[torch.dtype] = None,
         device: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    dtype_val: torch.dtype = dtype_from_str(dtype)
    dev = default_device(device)
    return torch.ones(shape, dtype=dtype_val, device=dev_from_str(dev))


def full_like(x: torch.Tensor,
              fill_value: Union[int, float],
              dtype: Optional[Union[torch.dtype, str]] = None,
              device: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    if device is None:
        device = _callable_dev(x)
    dtype = dtype_from_str(dtype)
    return torch.full_like(x, fill_value, dtype=dtype, device=default_device(device))


def ones_like(x : torch.Tensor,
              dtype: Optional[Union[torch.dtype, str]] = None,
              dev: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    if dev is None:
        dev = _callable_dev(x)
    dtype = dtype_from_str(dtype)
    return torch.ones_like(x, dtype= dtype, device=dev_from_str(dev))


def zeros_like(x: torch.Tensor,
               dtype: Optional[torch.dtype] = None,
               device: Optional[Union[torch.device, str]] = None)\
            -> torch.Tensor:
    if device is None:
        device = _callable_dev(x)
    if dtype is not None:
        return torch.zeros_like(x, dtype=dtype, device=dev_from_str(device))
    return torch.zeros_like(x, device=dev_from_str(device))


def tril(x: torch.Tensor,
         k: int = 0) \
         -> torch.Tensor:
    return torch.tril(x, diagonal=k)


def triu(x: torch.Tensor,
         k: int = 0) \
         -> torch.Tensor:
    return torch.triu(x, diagonal=k)
    

def empty(shape: Union[int, Tuple[int]],
          dtype: Optional[torch.dtype] = None,
          device: Optional[torch.device] = None) \
        -> Tensor:
    return torch.empty(shape, dtype=dtype_from_str(default_dtype(dtype)), device=dev_from_str(default_device(device)))


def empty_like(x: torch.Tensor,
              dtype: Optional[Union[torch.dtype, str]] = None,
              dev: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    if dev is None:
        dev = _callable_dev(x)
    dtype = dtype_from_str(dtype)
    return torch.empty_like(x, dtype=dtype, device=dev_from_str(dev))


def _differentiable_linspace(start, stop, num, device):
    if num == 1:
        return torch.unsqueeze(start, 0)
    n_m_1 = num - 1
    increment = (stop - start) / n_m_1
    increment_tiled = increment.repeat(n_m_1)
    increments = increment_tiled * torch.linspace(1, n_m_1, n_m_1, device=device)
    res = torch.cat((torch.unsqueeze(torch.tensor(start), 0), start + increments), 0)
    return res


# noinspection PyUnboundLocalVariable,PyShadowingNames
def linspace(start, stop, num, axis=None, dev=None):
    num = num.detach().numpy().item() if isinstance(num, torch.Tensor) else num
    start_is_array = isinstance(start, torch.Tensor)
    stop_is_array = isinstance(stop, torch.Tensor)
    linspace_method = torch.linspace
    dev = default_device(dev)
    sos_shape = []
    if start_is_array:
        start_shape = list(start.shape)
        sos_shape = start_shape
        if num == 1:
            return start.unsqueeze(axis).to(dev_from_str(dev))
        start = start.reshape((-1,))
        linspace_method = _differentiable_linspace if start.requires_grad else torch.linspace
    if stop_is_array:
        stop_shape = list(stop.shape)
        sos_shape = stop_shape
        if num == 1:
            return torch.ones(stop_shape[:axis] + [1] + stop_shape[axis:], device=dev_from_str(dev)) * start
        stop = stop.reshape((-1,))
        linspace_method = _differentiable_linspace if stop.requires_grad else torch.linspace
    if start_is_array and stop_is_array:
        if num < start.shape[0]:
            start = start.unsqueeze(-1)
            stop = stop.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num-1)
            res = [start]
            res += [start + inc*i for i in range(1, num-1)]
            res.append(stop)
        else:
            res = [linspace_method(strt, stp, num, device=dev_from_str(dev)) for strt, stp in zip(start, stop)]
        torch.cat(res, -1).reshape(start_shape + [num])
    elif start_is_array and not stop_is_array:
        if num < start.shape[0]:
            start = start.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(torch.ones_like(start, device=dev_from_str(dev)) * stop)
        else:
            res = [linspace_method(strt, stop, num, device=dev_from_str(dev)) for strt in start]
    elif not start_is_array and stop_is_array:
        if num < stop.shape[0]:
            stop = stop.unsqueeze(-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [torch.ones_like(stop, device=dev_from_str(dev)) * start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [linspace_method(start, stp, num, device=dev_from_str(dev)) for stp in stop]
    else:
        return linspace_method(start, stop, num, device=dev_from_str(dev))
    res = torch.cat(res, -1).reshape(sos_shape + [num])
    if axis is not None:
        res = torch.transpose(res, axis, -1)
    return res.to(dev_from_str(dev))


def eye(n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None) \
        -> torch.Tensor:
    dtype = dtype_from_str(default_dtype(dtype))
    device = dev_from_str(default_device(device))
    if n_cols is None:
        n_cols = n_rows
    i = torch.eye(n_rows, n_cols, dtype=dtype, device=device)
    if k == 0:
        return i
    elif -n_rows < k < 0:
        return torch.concat([torch.zeros([-k, n_cols], dtype=dtype, device=device),
                             i[:n_rows + k]], 0)
    elif 0 < k < n_cols:
        return torch.concat([torch.zeros([n_rows, k], dtype=dtype, device=device),
                             i[:, :n_cols - k]], 1)
    else:
        return torch.zeros([n_rows, n_cols], dtype=dtype, device=device)


def meshgrid(*arrays: torch.Tensor, indexing='xy')\
        -> List[torch.Tensor]:
    return list(torch.meshgrid(*arrays, indexing=indexing))


# noinspection PyShadowingNames
def arange(stop: Number, start: Number = 0, step: Number = 1, dtype: Optional[str] = None,
           dev: Optional[str] = None):
    dev = default_device(dev)
    if dtype is not None:
        return torch.arange(start, stop, step=step, dtype=dtype_from_str(dtype), device=dev_from_str(dev))
    else:
        return torch.arange(start, stop, step=step, device=dev_from_str(dev))


def full(shape: Union[int, Tuple[int, ...]],
         fill_value: Union[int, float],
         dtype: Optional[torch.dtype] = None,
         device: Optional[torch.device] = None) \
        -> Tensor:
    return torch.full(
        shape_to_tuple(shape), fill_value, dtype=dtype_from_str(default_dtype(dtype, fill_value)),
        device=default_device(device))


def from_dlpack(x):
    return torch.utils.dlpack.from_dlpack(x)


# Extra #
# ------#

array = asarray


def logspace(start, stop, num, base=10., axis=None, dev=None):
    power_seq = linspace(start, stop, num, axis, default_device(dev))
    return base ** power_seq
