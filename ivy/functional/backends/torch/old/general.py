"""
Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
import numpy as np
torch_scatter = None
import math as _math
import torch as _torch
from operator import mul
from torch.types import Number
from functools import reduce as _reduce
from typing import List, Dict, Optional, Union


# local
from ivy.functional.ivy.old import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.torch.device import dev_from_str, _callable_dev


# API #
# ----#



def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('torch.', '').replace('uint', '').replace('int', '').replace('bfloat', '').replace(
        'float', ''))





def shape(x, as_tensor=False) -> Union[_torch.Tensor, List[int]]:
    return _torch.tensor(x.shape) if as_tensor else x.shape


def get_num_dims(x, as_tensor=False) -> Union[_torch.Tensor, int]:
    return _torch.tensor(len(x.shape)) if as_tensor else len(x.shape)


def minimum(x, y):
    x_val = _torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = _torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return _torch.min(x_val, y_val)


def maximum(x, y):
    x_val = _torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = _torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return _torch.max(x_val, y_val)


def clip(x, x_min, x_max):
    return _torch.clamp(x, x_min, x_max)


# noinspection PyShadowingBuiltins
def round(x):
    return _torch.round(x)


def floor(x):
    return _torch.floor(x)


# noinspection PyShadowingBuiltins
def abs(x):
    return _torch.abs(x)

def argmin(x, axis: int = 0):
    ret = _torch.argmin(x, axis)
    if ret.shape == ():
        return ret.reshape(-1)
    return ret


def cast(x, dtype_in: str):
    dtype_val = dtype_from_str(dtype_in)
    return x.type(dtype_val)


astype = cast


# noinspection PyShadowingNames
def arange(stop: Number, start: Number = 0, step: Number = 1, dtype: Optional[str] = None,
           dev: Optional[str] = None):
    dev = default_device(dev)
    if dtype is not None:
        return _torch.arange(start, stop, step=step, dtype=dtype_from_str(dtype), device=dev_from_str(dev))
    else:
        return _torch.arange(start, stop, step=step, device=dev_from_str(dev))





def concatenate(xs: List[_torch.Tensor], axis: int = -1):
    if xs[0].shape == ():
        return _torch.cat([x.unsqueeze(0) for x in xs], axis)
    return _torch.cat(xs, axis)


def stack(xs: List[_torch.Tensor], axis: int = 0):
    return _torch.stack(xs, axis)



def split(x, num_or_size_splits: Optional[Union[int, List[int]]] = None, axis: int = 0, with_remainder: bool = False)\
        -> List[_torch.Tensor]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_or_size_splits))
        return [x]
    dim_size: int = x.shape[axis]
    if num_or_size_splits is None:
        # noinspection PyUnboundLocalVariable
        num_or_size_splits = 1
    elif isinstance(num_or_size_splits, int):
        if with_remainder:
            num_chunks = x.shape[axis] / num_or_size_splits
            num_chunks_int = _math.floor(num_chunks)
            remainder = num_chunks - num_chunks_int
            if remainder == 0:
                num_or_size_splits = _torch.round(_torch.tensor(dim_size) / _torch.tensor(num_or_size_splits))
            else:
                num_or_size_splits = tuple([num_or_size_splits] * num_chunks_int + [int(remainder*num_or_size_splits)])
        else:
            num_or_size_splits = _torch.round(_torch.tensor(dim_size) / _torch.tensor(num_or_size_splits))
    elif isinstance(num_or_size_splits, list):
        num_or_size_splits = tuple(num_or_size_splits)
    return list(_torch.split(x, num_or_size_splits, axis))


def repeat(x, repeats: Union[int, List[int]], axis: int = None):
    if len(x.shape) == 0 and axis in [0, -1]:
        axis = None
    return _torch.repeat_interleave(x, repeats, axis)


def tile(x, reps):
    if isinstance(reps, _torch.Tensor):
        reps = reps.detach().cpu().numpy().tolist()
    return x.repeat(reps)


# noinspection PyUnresolvedReferences
def constant_pad(x, pad_width: List[List[int]], value: Number = 0.):
    if x.shape == ():
        x = x.unsqueeze(0)
    if isinstance(pad_width, _torch.Tensor):
        pad_width = pad_width.detach().cpu().numpy().tolist()
    pad_width.reverse()
    pad_width_flat: List[int] = list()
    for pad_width_sec in pad_width:
        for item in pad_width_sec:
            pad_width_flat.append(item)
    return _torch.nn.functional.pad(x, pad_width_flat, mode='constant', value=value)


def zero_pad(x, pad_width: List[List[int]]):
    return constant_pad(x, pad_width, 0.)


def swapaxes(x, axis0: int, axis1: int):
    return _torch.transpose(x, axis0, axis1)


def transpose(x, axes: List[int]):
    if axes is None:
        num_dims = len(x.shape)
        axes = list(range(num_dims))
        axes.reverse()
    return x.permute(axes)


def where(condition, x1, x2):
    return _torch.where(condition.type(_torch.bool), x1, x2)


def indices_where(x):
    where_x = _torch.where(x)
    res = _torch.cat([_torch.unsqueeze(item, -1) for item in where_x], -1)
    return res


def reshape(x, newshape: List[int]):
    if isinstance(newshape, int):
        newshape = [newshape]
    return _torch.reshape(x, newshape)


def broadcast_to(x, new_shape):
    return x.expand(new_shape)


def squeeze(x, axis: Optional[int] = None):
    if axis is None:
        return _torch.squeeze(x)
    return _torch.squeeze(x, axis)




# noinspection PyShadowingNames
def zeros_like(x, dtype: Optional[str] = None, dev: Optional[str] = None):
    if dev is None:
        dev = _callable_dev(x)
    if dtype is not None:
        type_dict: Dict[str, _torch.dtype] = {'int8': _torch.int8,
            'int16': _torch.int16,
            'int32': _torch.int32,
            'int64': _torch.int64,
            'uint8': _torch.uint8,
            'bfloat16': _torch.bfloat16,
            'float16': _torch.float16,
            'float32': _torch.float32,
            'float64': _torch.float64,
            'bool': _torch.bool}
        return _torch.zeros_like(x, dtype=type_dict[dtype], device=dev_from_str(dev))
    return _torch.zeros_like(x, device=dev_from_str(dev))


def full(shape, fill_value, dtype=None, device=None):
    return _torch.full(
        ivy.shape_to_tuple(shape), fill_value, dtype=dtype_from_str(default_dtype(dtype, fill_value)),
        device=default_device(device))


# noinspection PyUnresolvedReferences,PyShadowingNames
def one_hot(indices, depth: int, dev: Optional[str] = None):
    if dev is None:
        dev = _callable_dev(indices)
    return _torch.nn.functional.one_hot(indices.type(_torch.int64), depth).to(dev_from_str(dev))


def cross(x1, x2):
    return _torch.cross(x1, x2)

def cumsum(x, axis: int = 0):
    return _torch.cumsum(x, axis)


def cumprod(x, axis: int = 0, exclusive: bool = False):
    if exclusive:
        x = _torch.transpose(x, axis, -1)
        x = _torch.cat((_torch.ones_like(x[..., -1:]), x[..., :-1]), -1)
        res = _torch.cumprod(x, -1)
        return _torch.transpose(res, axis, -1)
    return _torch.cumprod(x, axis)


# noinspection PyShadowingNames
def identity(n: int, dtype: ivy.Dtype = 'float32', batch_shape: Optional[List[int]] = None,
             dev: Optional[str] = None):
    dev = default_device(dev)
    type_dict: Dict[str, _torch.dtype] = {'int8': _torch.int8,
            'int16': _torch.int16,
            'int32': _torch.int32,
            'int64': _torch.int64,
            'uint8': _torch.uint8,
            'bfloat16': _torch.bfloat16,
            'float16': _torch.float16,
            'float32': _torch.float32,
            'float64': _torch.float64,
            'bool': _torch.bool}
    dtype_val: _torch.dtype = type_dict[dtype]
    mat = _torch.eye(n, n, dtype=dtype_val, device=dev_from_str(dev))
    if batch_shape is None:
        return mat
    else:
        reshape_dims = [1] * len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        res = _torch.reshape(mat, reshape_dims).repeat(tile_dims)
        return res


def meshgrid(*xs, indexing='ij'):
    ret = _torch.meshgrid(*xs)
    if indexing == 'xy':
        # ToDo: verify if this is correct
        return tuple([_torch.transpose(x, 1, 0) for x in ret])
    return ret


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size: Optional[int] = None, tensor: Optional[_torch.Tensor] = None,
                 reduction: str = 'sum', dev: Optional[str] = None):
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    if dev is None:
        dev = _callable_dev(updates)
    dtype = updates.dtype
    if reduction in ['sum', 'replace']:
        initial_val = _torch.tensor(0).type(dtype).to(dev_from_str(dev))
    elif reduction == 'min':
        initial_val = _torch.tensor(1e12).type(dtype).to(dev_from_str(dev))
    elif reduction == 'max':
        initial_val = _torch.tensor(-1e12).type(dtype).to(dev_from_str(dev))
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    if target_given:
        output = tensor
    else:
        output = _torch.ones([size], dtype=dtype).to(dev_from_str(dev)) * initial_val
    global torch_scatter
    if torch_scatter is None:
        try:
            import torch_scatter as torch_scatter
        except:
            raise Exception('Unable to import torch_scatter, verify this is correctly installed.')
    if reduction == 'replace':
        output[indices.type(_torch.int64)] = updates
        res = output
    else:
        res = torch_scatter.scatter(updates, indices.type(_torch.int64), out=output, reduce=reduction)
    if not target_given:
        return _torch.where(res == initial_val, _torch.zeros([size], dtype=updates.dtype).to(dev_from_str(dev)), res)
    return res


def _parse_ellipsis(so, ndims):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    return tuple(
        pre +
        [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))] +
        list(reversed(post))
    )


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape=None, tensor=None, reduction='sum', dev=None):

    # handle numeric updates
    updates = _torch.tensor([updates] if isinstance(updates, (float, int, bool)) else updates,
                            dtype=ivy.dtype(tensor, as_str=False) if ivy.exists(tensor)
                            else ivy.default_dtype(item=updates))

    # hanle non-tensor indices
    if indices == ():
        return updates
    elif indices is Ellipsis or (isinstance(indices, tuple) and indices == (Ellipsis,)):
        if updates.shape == () and ivy.exists(tensor) and tensor.shape == ():
            return updates
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = _torch.concat([_torch.unsqueeze(g, -1) for g in _torch.meshgrid(*[_torch.range(0, s) for s in shape])], -1)
    elif isinstance(indices, (float, int, bool)):
        indices = (indices,)
    if isinstance(indices, tuple):
        shape = tensor.shape if ivy.exists(tensor) else updates.shape
        indices = _parse_ellipsis(indices, len(shape))
        indices = _torch.concat([_torch.unsqueeze(g, -1) for g in _torch.meshgrid(
            *[_torch.range(0, s) if idx is slice(None, None, None) else _torch.tensor(idx) % s
              for s, idx in zip(shape, indices)])], -1)

    # broadcast updates to indices
    if updates.shape == ():
        updates = _torch.broadcast_to(updates, indices.shape[:-1])

    # implementation
    target = tensor
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.shape_to_tuple(target.shape) == ivy.shape_to_tuple(shape)
    if dev is None:
        dev = _callable_dev(updates)
    shape = list(shape) if ivy.exists(shape) else list(tensor.shape)
    dtype = updates.dtype
    indices_shape = indices.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [1]
    result_dim_sizes = _torch.tensor(result_dim_sizes_list).to(dev_from_str(dev))
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_result_size = _reduce(mul, shape, 1)
    if reduction in ['sum', 'replace']:
        initial_val = _torch.tensor(0).type(dtype).to(dev_from_str(dev))
    elif reduction == 'min':
        initial_val = _torch.tensor(1e12).type(dtype).to(dev_from_str(dev))
    elif reduction == 'max':
        initial_val = _torch.tensor(-1e12).type(dtype).to(dev_from_str(dev))
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    if target_given:
        flat_output = _torch.reshape(tensor, (flat_result_size,))
    else:
        flat_output = _torch.ones(flat_result_size, dtype=dtype).to(dev_from_str(dev)) * initial_val
    flat_updates = _torch.reshape(updates, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = _torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _torch.reshape(_torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)).repeat(
        *[1, implicit_indices_factor])
    implicit_indices = _torch.unsqueeze(_torch.arange(implicit_indices_factor).to(dev_from_str(dev)), 0).repeat(
        *[indices_for_flat_tiled.shape[0], 1])
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _torch.reshape(indices_for_flat, (-1,)).type(_torch.long)
    global torch_scatter
    if torch_scatter is None:
        try:
            import torch_scatter as torch_scatter
        except:
            raise Exception('Unable to import torch_scatter, verify this is correctly installed.')
    if reduction == 'replace':
        flat_output[flat_indices_for_flat] = flat_updates
        flat_scatter = flat_output
    else:
        flat_scatter = torch_scatter.scatter(flat_updates, flat_indices_for_flat, out=flat_output.clone(), reduce=reduction)
    if not target_given:
        # noinspection PyTypeChecker
        flat_scatter = _torch.where(flat_scatter == initial_val, _torch.zeros(flat_result_size, dtype=updates.dtype)
                                    .to(dev_from_str(dev)), flat_scatter)
    res = _torch.reshape(flat_scatter, list(shape))
    return res


# noinspection PyShadowingNames
def gather(params, indices, axis=-1, dev: Optional[str] = None):
    if dev is None:
        dev = _callable_dev(params)
    return _torch.gather(params, axis, indices.type(_torch.int64)).to(dev_from_str(dev))


# noinspection PyShadowingNames
def gather_nd(params, indices, dev: Optional[str] = None):
    if dev is None:
        dev = _callable_dev(params)
    indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(mul, params_shape[i + 1:], 1) for i in range(len(params_shape) - 1)] + [1]
    result_dim_sizes = _torch.tensor(result_dim_sizes_list).to(dev_from_str(dev))
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = _torch.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = _torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _torch.reshape(_torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)).repeat(
        *[1, implicit_indices_factor])
    implicit_indices = _torch.unsqueeze(_torch.arange(implicit_indices_factor).to(dev_from_str(dev)), 0).repeat(
        *[indices_for_flat_tiled.shape[0], 1])
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _torch.reshape(indices_for_flat, (-1,)).type(_torch.long)
    flat_gather = _torch.gather(flat_params, 0, flat_indices_for_flat)
    res = _torch.reshape(flat_gather, list(indices_shape[:-1]) + list(params_shape[num_index_dims:]))
    return res


def linear_resample(x, num_samples: int, axis: int = -1):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    num_vals = x_shape[axis]
    axis = axis % num_x_dims
    if axis != num_x_dims - 1:
        x_pre_shape = x_shape[0:axis] + x_shape[-1:] + x_shape[axis + 1:-1]
        x = _torch.swapaxes(x, axis, -1)
    else:
        x_pre_shape = x_shape[:-1]
    x = _torch.reshape(x, ([-1, 1] + [num_vals]))
    ret = _torch.nn.functional.interpolate(x, num_samples, mode='linear', align_corners=True)
    ret = _torch.reshape(ret, x_pre_shape + [num_samples])
    if axis != num_x_dims - 1:
        return _torch.transpose(ret, -1, axis)
    return ret


def dtype(x, as_str=False):
    dt = x.dtype
    if as_str:
        return dtype_to_str(dt)
    return dt


def dtype_to_str(dtype_in):
    if isinstance(dtype_in, str):
        return dtype_in
    return {_torch.int8: 'int8',
            _torch.int16: 'int16',
            _torch.int32: 'int32',
            _torch.int64: 'int64',
            _torch.uint8: 'uint8',
            _torch.bfloat16: 'bfloat16',
            _torch.float16: 'float16',
            _torch.float32: 'float32',
            _torch.float64: 'float64',
            _torch.bool: 'bool'}[dtype_in]


def dtype_from_str(dtype_in: str) -> _torch.dtype:
    if not isinstance(dtype_in, str):
        return dtype_in
    return {'int8': _torch.int8,
            'int16': _torch.int16,
            'int32': _torch.int32,
            'int64': _torch.int64,
            'uint8': _torch.uint8,
            'bfloat16': _torch.bfloat16,
            'float16': _torch.float16,
            'float32': _torch.float32,
            'float64': _torch.float64,
            'bool': _torch.bool}[dtype_in]


def compile(fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None):
    if dynamic:
        return _torch.jit.script(fn)
    return _torch.jit.trace(fn, example_inputs)


def current_framework_str():
    return 'torch'


def multiprocessing(context=None):
    import torch.multiprocessing
    if context is None:
        return torch.multiprocessing
    return torch.multiprocessing.get_context(context)


def container_types():
    return []


def inplace_update(x, val):
    x.data = val
    return x


def inplace_decrement(x, val):
    x.data -= val
    return x


def inplace_increment(x, val):
    x.data += val
    return x

inplace_arrays_supported = lambda: True
inplace_variables_supported = lambda: True
