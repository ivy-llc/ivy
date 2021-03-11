"""
Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch
import importlib
import numpy as np
torch_scatter = None
from operator import mul
from torch.types import Number
from functools import reduce as _reduce
from typing import List, Dict, Optional


# noinspection PyShadowingNames
def array(object_in, dtype_str: Optional[str] = None, dev: Optional[str] = None):
    if dev is not None:
        dev = dev.replace('gpu', 'cuda')
    if dtype_str is not None:
        return torch.tensor(object_in, dtype=dtype_from_str(dtype_str)).to(dev)
    else:
        return torch.tensor(object_in).to(dev)


def dtype_from_str(dtype_str_in: str) -> torch.dtype:
    return {'bool': torch.bool,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64}[dtype_str_in]


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise ValueError('Expected a pytroch tensor.')


def to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif torch.is_tensor(x):
        return x.detach().cpu().tolist()
    raise ValueError('Expected a pytroch tensor.')


def shape(x) -> List[int]:
    return x.shape


def get_num_dims(x) -> int:
    return len(x.shape)


def minimum(x, y):
    x_val = torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return torch.min(x_val, y_val)


def maximum(x, y):
    x_val = torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return torch.max(x_val, y_val)


def clip(x, x_min, x_max):
    return torch.clamp(x, x_min, x_max)


# noinspection PyShadowingBuiltins
def round(x):
    return torch.round(x)


def floormod(x, y):
    return x % y


def floor(x):
    return torch.floor(x)


def ceil(x):
    return torch.ceil(x)


# noinspection PyShadowingBuiltins
def abs(x):
    return torch.abs(x)


def argmax(x, axis: int = 0):
    return torch.argmax(x, axis)


def argmin(x, axis: int = 0):
    return torch.argmin(x, axis)


def cast(x, dtype_str_in: str):
    dtype_val = dtype_from_str(dtype_str_in)
    return x.type(dtype_val)


# noinspection PyShadowingNames
def arange(stop: Number, start: Number = 0, step: Number = 1, dtype_str: Optional[str] = None,
           dev: Optional[torch.device] = None):
    if dtype_str is not None:
        return torch.arange(start, stop, step=step, dtype=dtype_from_str(dtype_str)).to(dev)
    else:
        return torch.arange(start, stop, step=step).to(dev)


def _differentiable_linspace(start, stop, num):
    if num == 1:
        return torch.unsqueeze(start, 0)
    n_m_1 = num - 1
    increment = (stop - start) / n_m_1
    increment_tiled = increment.repeat(n_m_1)
    increments = increment_tiled * torch.linspace(1, n_m_1, n_m_1)
    res = torch.cat((torch.unsqueeze(start, 0), start + increments), 0)
    return res


# noinspection PyUnboundLocalVariable,PyShadowingNames
def linspace(start, stop, num, axis=None, dev=None):
    num = num.detach().numpy().item() if isinstance(num, torch.Tensor) else num
    start_is_array = isinstance(start, torch.Tensor)
    stop_is_array = isinstance(stop, torch.Tensor)
    linspace_method = torch.linspace
    if start_is_array:
        batch_shape = list(start.shape[:-1])
        start = start.reshape((-1,))
        linspace_method = _differentiable_linspace if start.requires_grad else torch.linspace
    if stop_is_array:
        batch_shape = list(stop.shape[:-1])
        stop = stop.reshape((-1,))
        linspace_method = _differentiable_linspace if stop.requires_grad else torch.linspace
    if start_is_array and stop_is_array:
        res = [linspace_method(strt, stp, num) for strt, stp in zip(start, stop)]
    elif start_is_array and not stop_is_array:
        res = [linspace_method(strt, stop, num) for strt in start]
    elif not start_is_array and stop_is_array:
        res = [linspace_method(start, stp, num) for stp in stop]
    else:
        return linspace_method(start, stop, num).to(dev)
    res = torch.cat(res, -1).reshape(batch_shape + [-1, num])
    if axis is not None:
        res = torch.transpose(res, axis, -1)
    return res.to(dev)


def concatenate(xs: List[torch.Tensor], axis: Optional[int] = None):
    if axis is None:
        xs = [torch.reshape(a, (-1,)) for a in xs]
        axis = 0
    return torch.cat(xs, axis)


def flip(x, axis: Optional[List[int]] = None, batch_shape: Optional[List[int]] = None):
    num_dims: int = len(batch_shape) if batch_shape is not None else len(x.shape)
    if axis is None:
        new_axis: List[int] = list(range(num_dims))
    else:
        new_axis: List[int] = axis
    if isinstance(new_axis, int):
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return torch.flip(x, new_axis)


def stack(xs: List[torch.Tensor], axis: int = 0):
    return torch.stack(xs, axis)


def unstack(x, axis: int, num_outputs: Optional[int] = None) -> List[torch.Tensor]:
    return list(torch.unbind(x, axis))


def split(x, num_sections: Optional[int] = None, axis: int = 0) -> List[torch.Tensor]:
    dim_size: int = x.shape[axis]
    if num_sections is None:
        # noinspection PyUnboundLocalVariable
        num_sections = dim_size
    chunk_size = round(torch.tensor(dim_size) / torch.tensor(num_sections))
    return list(torch.split(x, chunk_size, axis))


def tile(x, reps):
    return x.repeat(reps)


# noinspection PyUnresolvedReferences
def constant_pad(x, pad_width: List[List[int]], value: Number = 0., x_shape: Optional[List[int]] = None):
    pad_width.reverse()
    pad_width_flat: List[int] = list()
    for pad_width_sec in pad_width:
        for item in pad_width_sec:
            pad_width_flat.append(item)
    return torch.nn.functional.pad(x, pad_width_flat, mode='constant', value=value)


def zero_pad(x, pad_width: List[List[int]], x_shape: Optional[List[int]] = None):
    return constant_pad(x, pad_width, 0., x_shape)


def swapaxes(x, axis0, axis1):
    return torch.transpose(x, axis0, axis1)


def transpose(x, axes: List[int]):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return x.permute(axes)


def expand_dims(x, axis: int):
    return torch.unsqueeze(x, axis)


def where(condition, x1, x2, condition_shape: Optional[List[int]] = None, x_shape: Optional[List[int]] = None):
    return torch.where(condition, x1, x2)


def indices_where(x):
    where_x = torch.where(x)
    res = torch.cat([torch.unsqueeze(item, -1) for item in where_x], -1)
    return res


def reshape(x, newshape: List[int]):
    return torch.reshape(x, newshape)


def squeeze(x, axis: Optional[int] = None):
    if axis is None:
        return torch.squeeze(x)
    return torch.squeeze(x, axis)


# noinspection PyShadowingNames
def zeros(shape: List[int], dtype_str: str = 'float32', dev: Optional[torch.device] = None):
    type_dict: Dict[str, torch.dtype] = {'int32': torch.int32,
                                         'int64': torch.int64,
                                         'float32': torch.float32,
                                         'float64': torch.float64}
    dtype_val: torch.dtype = type_dict[dtype_str]
    return torch.zeros(shape, dtype=dtype_val).to(dev)


# noinspection PyShadowingNames
def zeros_like(x, dtype_str: Optional[str] = None, dev: Optional[torch.device] = None):
    if dev is None:
        dev = dev_str(x)
    if dtype_str is not None:
        type_dict: Dict[str, torch.dtype] = {'int32': torch.int32,
                                             'int64': torch.int64,
                                             'float32': torch.float32,
                                             'float64': torch.float64}
        return torch.zeros_like(x, dtype=type_dict[dtype_str]).to(dev)
    return torch.zeros_like(x).to(dev)


# noinspection PyShadowingNames
def ones(shape: List[int], dtype_str: str = 'float32', dev: Optional[str] = None):
    if dev is not None:
        dev: torch.device = dev.replace('gpu', 'cuda')
    type_dict: Dict[str, torch.dtype] = {'int32': torch.int32,
                                         'int64': torch.int64,
                                         'float32': torch.float32,
                                         'float64': torch.float64}
    dtype_val: torch.dtype = type_dict[dtype_str]
    return torch.ones(shape, dtype=dtype_val).to(dev)


# noinspection PyShadowingNames
def ones_like(x, dtype_str: Optional[str] = None, dev: Optional[torch.device] = None):
    if dev is None:
        dev = dev_str(x)
    if dtype_str is not None:
        type_dict: Dict[str, torch.dtype] = {'int32': torch.int32,
                                             'int64': torch.int64,
                                             'float32': torch.float32,
                                             'float64': torch.float64}
        return torch.ones_like(x, dtype=type_dict[dtype_str]).to(dev)
    return torch.ones_like(x).to(dev)


# noinspection PyUnresolvedReferences,PyShadowingNames
def one_hot(indices, depth: int, dev: Optional[torch.device] = None):
    if dev is None:
        dev = dev_str(indices)
    return torch.nn.functional.one_hot(indices, depth).to(dev)


def cross(x1, x2):
    return torch.cross(x1, x2)


def matmul(x1, x2, batch_shape: Optional[List[int]] = None):
    return torch.matmul(x1, x2)


def cumsum(x, axis: int = 0):
    return torch.cumsum(x, axis)


# noinspection PyShadowingNames
def identity(n: int, dtype_str: str = 'float32', batch_shape: Optional[List[int]] = None,
             dev: Optional[torch.device] = None):
    type_dict: Dict[str, torch.dtype] = {'int32': torch.int32,
                                         'int64': torch.int64,
                                         'float32': torch.float32,
                                         'float64': torch.float64}
    dtype_val: torch.dtype = type_dict[dtype_str]
    mat = torch.eye(n, n, dtype=dtype_val).to(dev)
    if batch_shape is None:
        return mat
    else:
        reshape_dims = [1] * len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        res = torch.reshape(mat, reshape_dims).repeat(tile_dims)
        return res


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size: int, reduction: str = 'sum', dev: Optional[torch.device] = None):
    if dev is None:
        dev = dev_str(updates)
    dtype = updates.dtype
    if reduction == 'sum':
        initial_val = torch.tensor(0).type(dtype).to(dev)
    elif reduction == 'min':
        initial_val = torch.tensor(1e12).type(dtype).to(dev)
    elif reduction == 'max':
        initial_val = torch.tensor(-1e12).type(dtype).to(dev)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    output = torch.ones([size], dtype=dtype).to(dev) * initial_val
    global torch_scatter
    if torch_scatter is None:
        try:
            import torch_scatter as torch_scatter
        except:
            raise Exception('Unable to import torch_scatter, verify this is correctly installed.')
    res = torch_scatter.scatter(updates, indices, out=output, reduce=reduction)
    res = torch.where(res == initial_val, torch.zeros([size], dtype=updates.dtype).to(dev), res)
    return res


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, num_idx_dims=None, reduction='sum', dev=None):
    if dev is None:
        dev = dev_str(updates)
    dev = dev.replace('gpu', 'cuda')
    shape = list(shape)
    dtype = updates.dtype
    indices_shape = indices.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [1]
    result_dim_sizes = torch.tensor(result_dim_sizes_list).to(dev)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_result_size = _reduce(mul, shape, 1)
    if reduction == 'sum':
        initial_val = torch.tensor(0).type(dtype).to(dev)
    elif reduction == 'min':
        initial_val = torch.tensor(1e12).type(dtype).to(dev)
    elif reduction == 'max':
        initial_val = torch.tensor(-1e12).type(dtype).to(dev)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    flat_output = torch.ones(flat_result_size, dtype=dtype).to(dev) * initial_val
    flat_updates = torch.reshape(updates, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = torch.reshape(torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)).repeat(
        *[1, implicit_indices_factor])
    implicit_indices = torch.unsqueeze(torch.arange(implicit_indices_factor).to(dev), 0).repeat(
        *[indices_for_flat_tiled.shape[0], 1])
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = torch.reshape(indices_for_flat, (-1,)).type(torch.long)
    global torch_scatter
    if torch_scatter is None:
        try:
            import torch_scatter as torch_scatter
        except:
            raise Exception('Unable to import torch_scatter, verify this is correctly installed.')
    flat_scatter = torch_scatter.scatter(flat_updates, flat_indices_for_flat, out=flat_output, reduce=reduction)
    # noinspection PyTypeChecker
    flat_scatter = torch.where(flat_scatter == initial_val, torch.zeros(flat_result_size, dtype=updates.dtype).to(dev),
                               flat_scatter)
    res = torch.reshape(flat_scatter, list(shape))
    return res


# noinspection PyShadowingNames
def gather_flat(params, indices, dev: Optional[torch.device] = None):
    if dev is None:
        dev = dev_str(params)
    return torch.gather(params, 0, indices)


# noinspection PyShadowingNames
def gather_nd(params, indices, indices_shape: Optional[List[int]] = None, dev: Optional[torch.device] = None):
    if dev is None:
        dev = dev_str(params)
    if indices_shape is None:
        indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(mul, params_shape[i + 1:], 1) for i in range(len(params_shape) - 1)] + [1]
    result_dim_sizes = torch.tensor(result_dim_sizes_list).to(dev)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = torch.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = torch.reshape(torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)).repeat(
        *[1, implicit_indices_factor])
    implicit_indices = torch.unsqueeze(torch.arange(implicit_indices_factor).to(dev), 0).repeat(
        *[indices_for_flat_tiled.shape[0], 1])
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = torch.reshape(indices_for_flat, (-1,)).type(torch.long)
    flat_gather = torch.gather(flat_params, 0, flat_indices_for_flat)
    res = torch.reshape(flat_gather, list(indices_shape[:-1]) + list(params_shape[num_index_dims:]))
    return res


def dev(x):
    return x.device


def dev_to_str(dev_in: torch.device):
    dev_type, dev_idx = (dev_in.type, dev_in.index)
    return dev_type + (':' + str(dev_idx) if dev_idx is not None else '')


def dev_str(x):
    return dev_to_str(dev(x))


gpu_is_available = torch.cuda.is_available


# noinspection PyUnresolvedReferences
def tpu_is_available():
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


def dtype(x):
    return x.dtype


def dtype_str(x):
    return {torch.int32: 'int32',
            torch.int64: 'int64',
            torch.float32: 'float32',
            torch.float64: 'float64'}[x.dtype]


def dtype_to_str(dtype_in):
    return {torch.int32: 'int32',
            torch.int64: 'int64',
            torch.float32: 'float32',
            torch.float64: 'float64'}[dtype_in]


def compile_fn(fn, dynamic=True, example_inputs=None):
    if dynamic:
        return torch.jit.script(fn)
    return torch.jit.trace(fn, example_inputs)
