"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
from numbers import Number
from operator import mul as _mul
from functools import reduce as _reduce
from tensorflow.python.types.core import Tensor

_round = round

# global
import tensorflow as _tf
import numpy as _np

DTYPE_DICT = {_tf.bool: 'bool',
              _tf.int8: 'int8',
              _tf.int16: 'int16',
              _tf.int32: 'int32',
              _tf.int64: 'int64',
              _tf.float16: 'float16',
              _tf.float32: 'float32',
              _tf.float64: 'float64'}


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev_str=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            tensor = _tf.convert_to_tensor(object_in)
            if dtype is None:
                return tensor
            return _tf.cast(tensor, dtype)
    else:
        tensor = _tf.convert_to_tensor(object_in)
        if dtype is None:
            return tensor
        return _tf.cast(tensor, dtype)


def is_array(x):
    if isinstance(x, Tensor):
        return True
    return False


to_numpy = lambda x: _np.asarray(_tf.convert_to_tensor(x))
to_list = lambda x: x.numpy().tolist()
shape = lambda x, as_tensor=False: _tf.shape(x) if as_tensor else tuple(x.shape)
get_num_dims = lambda x, as_tensor=False: _tf.shape(_tf.shape(x))[0] if as_tensor else int(_tf.shape(_tf.shape(x)))
minimum = _tf.minimum
maximum = _tf.maximum
clip = _tf.clip_by_value
# noinspection PyShadowingBuiltins
round = _tf.round
floormod = lambda x, y: x % y
floor = _tf.floor
ceil = _tf.math.ceil
# noinspection PyShadowingBuiltins
abs = _tf.abs
argmax = lambda x, axis=None: _tf.reshape(_tf.argmax(x, axis), get_num_dims(x))
argmin = lambda x, axis=None: _tf.reshape(_tf.argmin(x, axis), get_num_dims(x))


def cast(x, dtype_str):
    dtype_val = _tf.__dict__[dtype_str]
    return _tf.cast(x, dtype_val)


# noinspection PyShadowingNames
def arange(stop, start=0, step=1, dtype_str=None, dev_str=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            return _tf.range(start, stop, delta=step, dtype=dtype)
    else:
        return _tf.range(start, stop, delta=step, dtype=dtype)


def linspace(start, stop, num, axis=None, dev_str=None):
    if axis is None:
        axis = -1
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            return _tf.linspace(start, stop, num, axis=axis)
    else:
        return _tf.linspace(start, stop, num, axis=axis)


def concatenate(xs, axis=-1):
    if xs[0].shape == ():
        return _tf.concat([_tf.expand_dims(x, 0) for x in xs], axis)
    return _tf.concat(xs, axis)


def flip(x, axis=None, batch_shape=None):
    num_dims = len(batch_shape) if batch_shape is not None else len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return _tf.reverse(x, new_axis)


stack = _tf.stack


def unstack(x, axis):
    if x.shape == ():
        return [x]
    return _tf.unstack(x, axis=axis)


def split(x, num_sections=None, axis=0):
    if x.shape == ():
        if num_sections is not None and num_sections != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_sections))
        return [x]
    if num_sections is None:
        dim_size = _tf.shape(x)[axis]
        num_sections = dim_size
    return _tf.split(x, num_sections, axis)


def tile(x, reps):
    if x.shape == ():
        x = _tf.reshape(x, (-1,))
    if isinstance(reps, Number):
        reps = [reps]
    if isinstance(reps, Tensor) and reps.shape == ():
        reps = _tf.reshape(reps, (-1,))
    return _tf.tile(x, reps)


def constant_pad(x, pad_width, value=0):
    if x.shape == ():
        x = _tf.reshape(x, (-1,))
    return _tf.pad(x, pad_width, constant_values=value)


def zero_pad(x, pad_width):
    if x.shape == ():
        x = _tf.reshape(x, (-1,))
    return _tf.pad(x, pad_width)


def swapaxes(x, axis0, axis1):
    x_shape = x.shape
    num_dims = len(x_shape)
    axis0 %= num_dims
    axis1 %= num_dims
    config = list(range(num_dims))
    config.pop(axis0)
    config.insert(axis0, axis1)
    config.pop(axis1)
    config.insert(axis1, axis0)
    return _tf.transpose(x, config)


transpose = _tf.transpose
expand_dims = _tf.expand_dims
where = lambda condition, x1, x2: _tf.where(_tf.cast(condition, _tf.bool), x1, x2)
indices_where = _tf.where
reshape = _tf.reshape


def squeeze(x, axis=None):
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise Exception('tried to squeeze a zero-dimensional input by axis {}'.format(axis))
    return _tf.squeeze(x, axis)


# noinspection PyShadowingNames
def zeros(shape, dtype_str='float32', dev_str=None):
    dtype = _tf.__dict__[dtype_str]
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            return _tf.zeros(shape, dtype)
    else:
        return _tf.zeros(shape, dtype)


# noinspection PyShadowingNames
def zeros_like(x, dtype_str=None, dev_str=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            return _tf.zeros_like(x, dtype=dtype)
    else:
        return _tf.zeros_like(x, dtype=dtype)


# noinspection PyShadowingNames
def ones(shape, dtype_str='float32', dev_str=None):
    dtype = _tf.__dict__[dtype_str]
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            return _tf.ones(shape, dtype)
    else:
        return _tf.ones(shape, dtype)


# noinspection PyShadowingNames
def ones_like(x, dtype_str=None, dev_str=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            return _tf.ones_like(x, dtype=dtype)
    else:
        return _tf.ones_like(x, dtype=dtype)


def one_hot(indices, depth, dev_str=None):
    if dev_str is not None:
        with _tf.device('/' + dev_str.upper()):
            return _tf.one_hot(indices, depth)
    return _tf.one_hot(indices, depth)


cross = _tf.linalg.cross
matmul = lambda x1, x2: _tf.matmul(x1, x2)
cumsum = _tf.cumsum
cumprod = _tf.math.cumprod


# noinspection PyShadowingNames
def identity(n, dtype_str='float32', batch_shape=None, dev_str=None):
    dtype = _tf.__dict__[dtype_str]
    if dev_str:
        with _tf.device('/' + dev_str.upper()):
            return _tf.eye(n, n, batch_shape=batch_shape, dtype=dtype)
    else:
        return _tf.eye(n, n, batch_shape=batch_shape, dtype=dtype)


TF_SCATTER_VAR = {}


# noinspection PyShadowingNames
def scatter_flat(indices, updates, size, reduction='sum', dev_str=None):
    if dev_str is None:
        dev_str = _dev_str_callable(updates)
    dtype = updates.dtype
    if reduction == 'sum':
        return _tf.scatter_nd(_tf.expand_dims(indices, -1), updates, [size])
    elif reduction == 'min':
        func = _tf.compat.v1.scatter_min
        initial_val = _tf.cast(_tf.constant(2 ** 31 - 1), dtype)
    elif reduction == 'max':
        func = _tf.compat.v1.scatter_max
        initial_val = _tf.cast(_tf.constant(-(2 ** 31 - 1)), dtype)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    global TF_SCATTER_VAR
    if size not in TF_SCATTER_VAR:
        TF_SCATTER_VAR[size] = {dtype: _tf.Variable(_tf.ones(size, dtype=dtype) * initial_val, trainable=False)}
    elif dtype not in TF_SCATTER_VAR[size]:
        TF_SCATTER_VAR[size][dtype] = _tf.Variable(_tf.ones(size, dtype=dtype) * initial_val, trainable=False)
    else:
        TF_SCATTER_VAR[size][dtype].assign(_tf.ones(size, dtype=dtype) * initial_val)
    res = _tf.convert_to_tensor(func(TF_SCATTER_VAR[size][dtype], indices, updates))
    res = _tf.where(res == initial_val, _tf.zeros(size, dtype=updates.dtype), res)
    with _tf.device('/' + dev_str.upper()):
        return res


# noinspection PyShadowingNames
def scatter_nd(indices, updates, shape, reduction='sum', dev_str=None):
    if dev_str is None:
        dev_str = _dev_str_callable(updates)
    shape = list(shape)
    dtype = updates.dtype
    if reduction == 'sum':
        return _tf.scatter_nd(indices, updates, shape)
    elif reduction == 'min':
        func = _tf.compat.v1.scatter_min
        initial_val = _tf.cast(_tf.constant(2 ** 31 - 1), dtype)
    elif reduction == 'max':
        func = _tf.compat.v1.scatter_max
        initial_val = _tf.cast(_tf.constant(-(2 ** 31 - 1)), dtype)
    else:
        raise Exception('reduction is {}, but it must be one of "sum", "min" or "max"'.format(reduction))
    indices_shape = indices.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [_reduce(_mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [1]
    result_dim_sizes = _tf.constant(result_dim_sizes_list)
    implicit_indices_factor = result_dim_sizes[num_index_dims - 1]
    flat_result_size = _reduce(_mul, shape, 1)
    global TF_SCATTER_VAR
    if flat_result_size not in TF_SCATTER_VAR:
        TF_SCATTER_VAR[flat_result_size] = {dtype: _tf.Variable(_tf.ones(flat_result_size, dtype=dtype) * initial_val, trainable=False)}
    elif dtype not in TF_SCATTER_VAR[flat_result_size]:
        TF_SCATTER_VAR[flat_result_size][dtype] = _tf.Variable(_tf.ones(flat_result_size, dtype=dtype) * initial_val, trainable=False)
    else:
        TF_SCATTER_VAR[flat_result_size][dtype].assign(_tf.ones(flat_result_size, dtype=dtype) * initial_val)
    flat_updates = _tf.reshape(updates, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = _tf.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = _tf.tile(_tf.reshape(_tf.reduce_sum(indices * indices_scales, -1, keepdims=True), (-1, 1)), [1, implicit_indices_factor])
    implicit_indices = _tf.tile(_tf.expand_dims(_tf.range(implicit_indices_factor), 0), _tf.stack((_tf.shape(indices_for_flat_tiled)[0], _tf.constant(1))))
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = _tf.reshape(indices_for_flat, (-1,))
    flat_scatter = _tf.convert_to_tensor(func(TF_SCATTER_VAR[flat_result_size][dtype], flat_indices_for_flat, flat_updates))
    flat_scatter = _tf.where(flat_scatter == initial_val, _tf.zeros(flat_result_size, dtype=updates.dtype), flat_scatter)
    with _tf.device('/' + dev_str.upper()):
        res = _tf.reshape(flat_scatter, list(shape))
        return res


def gather(params, indices, axis=-1, dev_str=None):
    axis = axis % len(indices.shape)
    if dev_str is None:
        dev_str = _dev_str_callable(params)
    with _tf.device('/' + dev_str.upper()):
        return _tf.gather(params, indices, axis=axis, batch_dims=axis)


def gather_nd(params, indices, dev_str=None):
    if dev_str is None:
        dev_str = _dev_str_callable(params)
    with _tf.device('/' + dev_str.upper()):
        return _tf.gather_nd(params, indices)


dev = lambda x: x.device


def dev_to_str(dev_in):
    return ':'.join(dev_in.split(':')[-2:]).lower()


dev_str = lambda x: dev_to_str(dev(x))
_dev_str_callable = dev_str
gpu_is_available = lambda: len(_tf.config.list_physical_devices('GPU')) > 0


def tpu_is_available():
    try:
        resolver = _tf.distribute.cluster_resolver.TPUClusterResolver()
        _tf.config.experimental_connect_to_cluster(resolver)
        _tf.tpu.experimental.initialize_tpu_system(resolver)
        _tf.config.list_logical_devices('TPU')
        _tf.distribute.experimental.TPUStrategy(resolver)
        return True
    except ValueError:
        return False


dtype = lambda x: x.dtype
dtype_str = lambda x: DTYPE_DICT[x.dtype]
dtype_to_str = lambda dtype_in: DTYPE_DICT[dtype_in]
compile_fn = lambda fn, dynamic=True, example_inputs=None: _tf.function(fn)
