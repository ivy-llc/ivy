"""
Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature.
"""

# global
from functools import reduce as _reduce
_torch_scatter = None
from operator import mul as _mul
import torch as _torch
import numpy as _np
_round = round


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None):
    dtype = _torch.__dict__[dtype_str] if dtype_str else dtype_str
    return _torch.tensor(object_in, dtype=dtype).to(dev)


def to_numpy(x):
    if isinstance(x, _np.ndarray):
        return x
    elif _torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise ValueError('Expected a pytroch tensor.')


def to_list(x):
    if isinstance(x, _np.ndarray):
        return x.tolist()
    elif _torch.is_tensor(x):
        return x.detach().cpu().tolist()
    raise ValueError('Expected a pytroch tensor.')


shape = lambda x: x.shape
get_num_dims = lambda x: len(x.shape)


def minimum(x, y):
    x_val = _torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = _torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return _torch.min(x_val, y_val)


def maximum(x, y):
    x_val = _torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = _torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return _torch.max(x_val, y_val)


clip = _torch.clamp
# noinspection PyShadowingBuiltins
round = _torch.round
floormod = lambda x, y: x % y
floor = _torch.floor
ceil = _torch.ceil
# noinspection PyShadowingBuiltins
abs = _torch.abs
argmax = lambda x, axis=0: _torch.argmax(x, axis)
argmin = lambda x, axis=0: _torch.argmin(x, axis)


def cast(x, dtype_str):
    dtype_val = _torch.__dict__[dtype_str]
    return x.type(dtype_val)


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype_str=None, dev=None):
    dtype = _torch.__dict__[dtype_str] if dtype_str else dtype_str
    return _torch.arange(start, stop, step=step, dtype=dtype).to(dev)


def _differentiable_linspace(start, stop, num):
    if num == 1:
        return _torch.unsqueeze(start, 0)
    n_m_1 = num - 1
    increment = (stop - start)/n_m_1
    increment_tiled = increment.repeat(n_m_1)
    increments = increment_tiled * _torch.linspace(1, n_m_1, n_m_1)
    res = _torch.cat((_torch.unsqueeze(start, 0), start + increments), 0)
    return res


def linspace(start, stop, num, axis=None, dev=None):
    num = num.detach().numpy().item() if isinstance(num, _torch.Tensor) else num
    start_is_array = isinstance(start, _torch.Tensor)
    stop_is_array = isinstance(stop, _torch.Tensor)
    linspace_method = _torch.linspace
    if start_is_array:
        batch_shape = list(start.shape[:-1])
        start = start.reshape((-1,))
        linspace_method = _differentiable_linspace if start.requires_grad else _torch.linspace
    if stop_is_array:
        batch_shape = list(stop.shape[:-1])
        stop = stop.reshape((-1,))
        linspace_method = _differentiable_linspace if stop.requires_grad else _torch.linspace
    if start_is_array and stop_is_array:
        res = [linspace_method(strt, stp, num) for strt, stp in zip(start, stop)]
    elif start_is_array and not stop_is_array:
        res = [linspace_method(strt, stop, num) for strt in start]
    elif not start_is_array and stop_is_array:
        res = [linspace_method(start, stp, num) for stp in stop]
    else:
        return linspace_method(start, stop, num).to(dev)
    res = _torch.cat(res, -1).reshape(batch_shape + [-1, num])
    if axis is not None:
        res = _torch.transpose(res, axis, -1)
    return res.to(dev)


def concatenate(xs, axis=None):
    if axis is None:
        xs = [_torch.reshape(a, (-1,)) for a in xs]
        axis = 0
    return _torch.cat(xs, axis)


def flip(x, axis=None, batch_shape=None):
    num_dims = len(batch_shape) if batch_shape is not None else len(x.shape)
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return _torch.flip(x, new_axis)


stack = lambda xs, axis=0: _torch.stack(xs, axis)
unstack = lambda x, axis, _=None: list(_torch.unbind(x, axis))


def split(x, num_sections=None, axis=0):
    dim_size = x.shape[axis]
    if num_sections is None:
        # noinspection PyUnboundLocalVariable
        num_sections = dim_size
    chunk_size = _round(dim_size/num_sections)
    return list(_torch.split(x, chunk_size, axis))


tile = lambda x, reps: x.repeat(reps)


def zero_pad(x, pad_width, _=None):
    pad_width.reverse()
    pad_width_flat = [item for sublist in pad_width for item in sublist]
    return _torch.nn.functional.pad(x, pad_width_flat, mode='constant')


swapaxes = _torch.transpose


def transpose(x, axes=None):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return x.permute(axes)


expand_dims = lambda x, axis: _torch.unsqueeze(x, axis)
where = lambda condition, x1, x2, _=None, _1=None: _torch.where(condition, x1, x2)


def indices_where(x):
    where_x = _torch.where(x)
    res = _torch.cat([_torch.unsqueeze(item, -1) for item in where_x], -1)
    return res


reshape = _torch.reshape


def squeeze(x, axis=None):
    args = [item for item in [x, axis] if item is not None]
    return _torch.squeeze(*args)


# noinspection PyShadowingNames
def zeros(shape, dtype_str='float32', dev=None):
    dtype_val = _torch.__dict__[dtype_str]
    return _torch.zeros(shape, dtype=dtype_val).to(dev)


# noinspection PyShadowingNames
def zeros_like(x, dtype_str=None, dev=None):
    dtype = _torch.__dict__[dtype_str] if dtype_str else dtype_str
    if dev is None:
        dev = get_device(x)
    return _torch.zeros_like(x, dtype=dtype).to(dev)


# noinspection PyShadowingNames
def ones(shape, dtype_str='float32', dev=None):
    dtype = _torch.__dict__[dtype_str] if dtype_str else dtype_str
    return _torch.ones(shape, dtype=dtype).to(dev)


# noinspection PyShadowingNames
def ones_like(x, dtype_str=None, dev=None):
    dtype = _torch.__dict__[dtype_str] if dtype_str else dtype_str
    if dev is None:
        dev = get_device(x)
    return _torch.ones_like(x, dtype=dtype).to(dev)


def one_hot(indices, depth, dev=None):
    if dev is None:
        dev = get_device(indices)
    return _torch.nn.functional.one_hot(indices, depth).to(dev)


cross = _torch.cross
matmul = lambda x1, x2, _=None: _torch.matmul(x1, x2)
cumsum = lambda x, axis=0: _torch.cumsum(x, axis)


# noinspection PyShadowingNames
def identity(n, dtype_str='float32', batch_shape=None, dev=None):
    dtype = _torch.__dict__[dtype_str] if dtype_str else dtype_str
    mat = _torch.eye(n, n, dtype=dtype).to(dev)
    if batch_shape is None:
        return mat
    else:
        reshape_dims = [1]*len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        res = _torch.reshape(mat, reshape_dims).repeat(tile_dims)
        return res


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size, reduction='sum', dev=None):
    if dev is None:
        dev = get_device(updates)
    dtype = updates.dtype
    if reduction == 'sum':
        initial_val = _torch.tensor(0).type(dtype).to(dev)
    elif reduction == 'min':
        initial_val = _torch.tensor(1e12).type(dtype).to(dev)
    elif reduction == 'max':
        initial_val = _torch.tensor(-1e12).type(dtype).to(dev)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    output = _torch.ones([size], dtype=dtype).to(dev) * initial_val
    global _torch_scatter
    if _torch_scatter is None:
        try:
            import torch_scatter as _torch_scatter
        except:
            raise Exception('Unable to import torch_scatter, verify this is correctly installed.')
    res = _torch_scatter.scatter(updates, indices, out=output, reduce=reduction)
    res = _torch.where(res == initial_val, _torch.zeros([size], dtype=updates.dtype).to(dev), res)
    return res


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, num_idx_dims=None, reduction='sum', dev=None):
    if dev is None:
        dev = get_device(updates)
    shape = list(shape)
    dtype = updates.dtype
    indices_shape = indices.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(_mul, shape[i+1:], 1) for i in range(len(shape)-1)] + [1]
    result_dim_sizes = _torch.tensor(result_dim_sizes_list).to(dev)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_result_size = _reduce(_mul, shape, 1)
    if reduction == 'sum':
        initial_val = _torch.tensor(0).type(dtype).to(dev)
    elif reduction == 'min':
        initial_val = _torch.tensor(1e12).type(dtype).to(dev)
    elif reduction == 'max':
        initial_val = _torch.tensor(-1e12).type(dtype).to(dev)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    flat_output = _torch.ones(flat_result_size, dtype=dtype).to(dev) * initial_val
    flat_updates = _torch.reshape(updates, (-1,))
    new_shape = [1]*(len(indices_shape)-1) + [num_index_dims]
    indices_scales = _torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _torch.reshape(_torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)).repeat(*[1, implicit_indices_factor])
    implicit_indices = _torch.unsqueeze(_torch.arange(implicit_indices_factor).to(dev), 0).repeat(*[indices_for_flat_tiled.shape[0], 1])
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _torch.reshape(indices_for_flat, (-1,)).type(_torch.long)
    global _torch_scatter
    if _torch_scatter is None:
        try:
            import torch_scatter as _torch_scatter
        except:
            raise Exception('Unable to import torch_scatter, verify this is correctly installed.')
    flat_scatter = _torch_scatter.scatter(flat_updates, flat_indices_for_flat, out=flat_output, reduce=reduction)
    # noinspection PyTypeChecker
    flat_scatter = _torch.where(flat_scatter == initial_val, _torch.zeros(flat_result_size, dtype=updates.dtype).to(dev), flat_scatter)
    res = _torch.reshape(flat_scatter, list(shape))
    return res


def gather_flat(params, indices, dev=None):
    if dev is None:
        dev = get_device(params)
    return _torch.gather(params, 0, indices)


def gather_nd(params, indices, indices_shape=None, dev=None):
    if dev is None:
        dev = get_device(params)
    if indices_shape is None:
        indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(_mul, params_shape[i + 1:], 1) for i in range(len(params_shape) - 1)] + [1]
    result_dim_sizes = _torch.tensor(result_dim_sizes_list).to(dev)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = _torch.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = _torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _torch.reshape(_torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)).repeat(*[1, implicit_indices_factor])
    implicit_indices = _torch.unsqueeze(_torch.arange(implicit_indices_factor).to(dev), 0).repeat(*[indices_for_flat_tiled.shape[0], 1])
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _torch.reshape(indices_for_flat, (-1,)).type(_torch.long)
    flat_gather = _torch.gather(flat_params, 0, flat_indices_for_flat)
    res = _torch.reshape(flat_gather, list(indices_shape[:-1]) + list(params_shape[num_index_dims:]))
    return res


def get_device(x):
    dev_type, dev_idx = (x.device.type, x.device.index)
    return dev_type + (':' + str(dev_idx) if dev_idx is not None else '')


dtype = lambda x: x.dtype
compile_fn = lambda fn, example_inputs: _torch.jit.trace(fn, example_inputs)
