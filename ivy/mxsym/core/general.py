"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import time as _time
import mxnet as _mx
import numpy as _np
_round = round
import logging


def _raise(ex):
    raise ex


def _mxnet_init_context(dev):
    if dev is None or dev.find("cpu") != -1:
        mx_dev = "cpu"
    elif dev.find("cuda") != -1:
        mx_dev = "gpu"
    else:
        raise Exception("dev type not supported.")
    if dev.find(":") != -1:
        mx_dev_id = int(dev[dev.find(":")+1:])
    else:
        mx_dev_id = 0
    return _mx.Context(mx_dev, mx_dev_id)


def array(object_in, dtype_str=None, dev=None):
    if isinstance(object_in, _mx.symbol.symbol.Symbol):
        return object_in
    _mx_nd_array = _mx.nd.array(object_in)
    return _mx.sym.BlockGrad(_mx.symbol.Variable(str(_time.time()).replace('.', ''), shape=_mx_nd_array.shape,
                                                 dtype=dtype_str, init=_mx.init.Constant(value=_mx_nd_array)))


def to_numpy(_):
    raise Exception('MXNet Symbolic mode does not support to_numpy().')


def to_list(_):
    raise Exception('MXNet Symbolic mode does not support to_list().')


shape = lambda x: _mx.symbol.shape_array(x)
get_num_dims = lambda x: _mx.symbol.shape_array((_mx.symbol.shape_array(x)))
minimum = _mx.symbol.minimum
maximum = _mx.symbol.maximum
clip = _mx.symbol.clip
# noinspection PyShadowingBuiltins
round = _mx.symbol.round
floormod = lambda x, y: x % y
floor = _mx.symbol.floor
ceil = _mx.symbol.ceil
# noinspection PyShadowingBuiltins
abs = _mx.symbol.abs
argmax = lambda x, axis=0: _mx.symbol.argmax(x, axis)
argmin = lambda x, axis=0: _mx.symbol.argmin(x, axis)
cast = lambda x, dtype_str: x.astype(dtype_str)
arange = lambda stop, start=0, step=1, dtype_str=None, dev=None: \
    _mx.symbol.arange(start, stop, step=step, dtype=dtype_str)
linspace = lambda _, _1, _2, _3=None, _4=None: _raise(Exception('MXNet does not support linspace().'))


def concatenate(xs, axis=None):
    if axis is None:
        xs = [_mx.symbol.reshape(a, (-1,)) for a in xs]
        axis = 0
    return _mx.symbol.concat(*xs, dim=axis)


def flip(x, axis=None, batch_shape=None):
    num_dims = len(batch_shape) if batch_shape is not None else len(_mx.symbol.shape_array(x))
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return _mx.symbol.flip(x, new_axis)


stack = lambda xs, axis=0: _mx.symbol.stack(*xs, axis=axis)


def unstack(x, axis, num_outputs=None):
    num_outputs = _mx.symbol.shape_array(x)[axis] if not num_outputs else num_outputs
    return _mx.symbol.split(x, num_outputs, axis, squeeze_axis=True)


def split(x, num_sections=None, axis=0):
    num_sections = _mx.symbol.shape_array(x)[axis] if not num_sections else num_sections
    return _mx.symbol.split(x, num_sections, axis)


tile = _mx.symbol.tile


def zero_pad(x, pad_width, x_shape=None):
    x_shape = list(_mx.symbol.shape_array(x)) if not x_shape else x_shape
    num_dims = len(x_shape)
    if num_dims > 3:
        raise Exception('Invalid inputs. Pad for mxnet only supports inputs with 3 dimensions or smaller.')
    num_dims_to_add = 4 - num_dims
    mat_expanded_dims = _mx.symbol.reshape(x, tuple([1] * num_dims_to_add + x_shape))
    pad_width_flat = [0]*num_dims_to_add*2 + [item for sublist in pad_width for item in sublist]
    pad_expanded_dims = _mx.symbol.pad(mat_expanded_dims, mode="constant", pad_width=tuple(pad_width_flat))
    new_shape = [orig_dim + pad_width_item[0] + pad_width_item[1]
                 for orig_dim, pad_width_item in zip(x_shape, pad_width)]
    return _mx.symbol.reshape(pad_expanded_dims, tuple(new_shape))


swapaxes = _mx.symbol.swapaxes


def transpose(x, axes=None):
    if axes is None:
        num_dims = len(_mx.symbol.shape_array(x))
        axes = list(range(num_dims))
        axes.reverse()
    return _mx.symbol.transpose(x, axes)


expand_dims = _mx.symbol.expand_dims


def where(condition, x1, x2, condition_shape=None, x_shape=None):
    if x_shape is None or condition_shape is None:
        raise Exception('x_shape and condition_shape must be provided for calling ivy.where()'
                        'in mxnet symbolic mode.')
    if x_shape == condition_shape:
        return _mx.symbol.where(condition, x1, x2)
    tile_reps = [int(x / c) for x, c in zip(x_shape, condition_shape)]
    tiled_condition = _mx.symbol.tile(condition, tile_reps)
    return _mx.symbol.where(tiled_condition, x1, x2)


indices_where = lambda _: _raise(Exception('MXNet does not support indices_where function.'))
reshape = _mx.symbol.reshape
squeeze = lambda x, axis=None: _mx.symbol.squeeze(x, axis)


# noinspection PyShadowingNames
def zeros(shape, dtype_str='float32', dev=None):
    cont = _mxnet_init_context('cpu' if not dev else dev)
    return _mx.symbol.zeros(shape, ctx=cont).astype(dtype_str)


def zeros_like(x, dtype_str=None, dev=None):
    mx_zeros = _mx.symbol.zeros_like(x, ctx=_mxnet_init_context('cpu' if not dev else dev))
    return mx_zeros if not dtype_str else mx_zeros.astype(dtype_str)


# noinspection PyShadowingNames
def ones(shape, dtype_str='float32', dev=None):
    cont = _mxnet_init_context('cpu' if not dev else dev)
    return _mx.symbol.ones(shape, ctx=cont).astype(dtype_str)


def ones_like(x, dtype_str=None, dev=None):
    mx_ones = _mx.symbol.ones_like(x, ctx=_mxnet_init_context('cpu' if not dev else dev))
    return mx_ones if dtype_str is None else mx_ones.astype(dtype_str)


# noinspection PyUnusedLocal
def one_hot(indices, depth, dev=None):
    return _mx.symbol.one_hot(indices, depth)


def cross(x1, x2):
    raise Exception('MXNet Symbolic mode does not support array slicing, and so does not support cross().')


def matmul(x1, x2, batch_shape=None):
    expand = len(batch_shape) == 0
    if expand:
        x1 = _mx.symbol.expand_dims(x1, 0)
        x2 = _mx.symbol.expand_dims(x2, 0)
    res = _mx.symbol.batch_dot(x1, x2)
    return _mx.symbol.squeeze(res, 0) if expand else res


cumsum = lambda x, axis=0: _mx.symbol.cumsum(x, axis)


def identity(n, dtype_str='float32', batch_shape=None, dev=None):
    mat = _mx.symbol.eye(n, dtype=dtype_str)
    if batch_shape is None:
        return mat
    else:
        reshape_dims = [1]*len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        return _mx.symbol.tile(_mx.symbol.reshape(mat, reshape_dims), tile_dims)


def scatter_flat(indices, updates, size, reduction='sum', dev=None):
    if reduction == 'sum':
        return _mx.symbol.scatter_nd(updates, _mx.symbol.expand_dims(indices, 0), [size])
    else:
        raise Exception('MXNet scatter_nd currently only supports reduction mode "sum", but {} selected.'.
                        format(reduction))


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, num_idx_dims, reduction='sum', dev=None):
    shape = list(shape)
    transpose_order = [num_idx_dims-1] + list(range(num_idx_dims-1))
    indices = _mx.symbol.transpose(indices, transpose_order)
    shape = shape if type(shape) is list else shape.asnumpy().astype(_np.int32).tolist()
    if reduction == 'sum':
        return _mx.symbol.scatter_nd(updates, indices, shape)
    else:
        raise Exception('MXNet scatter_nd currently only supports reduction mode "sum", but {} selected.'.
                        format(reduction))


def gather_flat(params, indices, dev=None):
    return _mx.symbol.gather_nd(params, _mx.symbol.expand_dims(indices, 0))


def gather_nd(params, indices, indices_shape=None, dev=None):
    if indices_shape is None:
        indices_shape = _mx.symbol.shape_array(indices)
    num_idx_dims = len(indices_shape)
    transpose_order = [num_idx_dims-1] + list(range(num_idx_dims-1))
    indices = _mx.symbol.transpose(indices, transpose_order)
    return _mx.symbol.gather_nd(params, indices)


get_device = lambda _: _raise(Exception('mxnet symbolic tensors do not have a context'))
dtype = lambda _: _raise(Exception('MXNet Symbolic mode does not support dtype().'))


# noinspection PyUnusedLocal
def compile_fn(func, example_inputs=None):
    logging.warning('MXnet does not support compiling arbitrary functions, '
                    'However, you are currently using the MXNet Symbolic backend, which does compile the functions.\n'
                    'Now returning the unmodified function.')
    return func
