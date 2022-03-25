# global
import mxnet as mx
from typing import Tuple, Union, Optional, Iterable

# local
from ivy import default_device, dtype_from_str, default_dtype, dtype_to_str
from ivy.functional.backends.mxnet import _mxnet_init_context
from ivy.functional.backends.mxnet import _1_dim_array_to_flat_array

def asarray(object_in, dtype: Optional[str] = None, dev: Optional[str] = None, copy: Optional[bool] = None):
    # mxnet don't have asarray implementation, haven't properly tested
    cont = _mxnet_init_context(default_device(dev))
    if copy is None:
        copy = False
    if copy:
        if dtype is None and isinstance(object_in, mx.nd.NDArray):
            return mx.nd.array(object_in, cont).as_in_context(cont)
        if dtype is None and not isinstance(object_in, mx.nd.NDArray):
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))
        else:
            dtype = dtype_to_str(default_dtype(dtype, object_in))
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))
    else:
        if dtype is None and isinstance(object_in, mx.nd.NDArray):
            return object_in.as_in_context(cont)
        if dtype is None and not isinstance(object_in, mx.nd.NDArray):
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))
        else:
            dtype = dtype_to_str(default_dtype(dtype, object_in))
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))


def zeros(shape: Union[int, Tuple[int]],
          dtype: Optional[type] = None,
          device: Optional[mx.context.Context] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.zeros((1,), ctx=cont).astype(dtype))
    return mx.nd.zeros(shape, ctx=cont).astype(dtype)


def ones(shape: Union[int, Tuple[int]],
         dtype: Optional[type] = None,
         device: Optional[str] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    shape = [shape] if shape is not isinstance(shape, Iterable) else shape
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.ones((1,), ctx=cont).astype(dtype))
    return mx.nd.ones(shape, ctx=cont).astype(dtype)


def ones_like(x : mx.ndarray.ndarray.NDArray,
              dtype : Optional[Union[type, str]] = None,
              dev : Optional[Union[mx.context.Context, str]] = None) \
        -> mx.ndarray.ndarray.NDArray:
    if x.shape == ():
        return mx.nd.array(1., ctx=_mxnet_init_context(default_device(dev)))
    mx_ones = mx.nd.ones_like(x, ctx=_mxnet_init_context(default_device(dev)))
    return mx_ones if dtype is None else mx_ones.astype(dtype)

  
def tril(x: mx.ndarray.ndarray.NDArray,
         k: int = 0) \
         -> mx.ndarray.ndarray.NDArray:
    return mx.np.tril(x, k)


def empty(shape: Union[int, Tuple[int]],
          dtype: Optional[type] = None,
          device: Optional[mx.context.Context] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    return mx.nd.empty(shape, dtype_from_str(default_dtype(dtype)), cont)


def _linspace(start, stop, num, cont):
    if num == 1:
        return start
    start = mx.nd.array(start).reshape((1,)).astype('float32')
    stop = mx.nd.array(stop).reshape((1,)).astype('float32')
    n_m_1 = mx.nd.array(num - 1).reshape((1,)).astype('float32')
    increment = (stop - start)/n_m_1
    increment_tiled = mx.nd.tile(increment, num - 1)
    increments = increment_tiled * mx.nd.array(mx.nd.np.linspace(1, num - 1, num - 1).tolist(), ctx=cont)
    ret = mx.nd.concat(start, start + increments, dim=0)
    return ret


def linspace(start, stop, num, axis=None, dev=None):
    cont = _mxnet_init_context(default_device(dev))
    num = num.asnumpy()[0] if isinstance(num, mx.nd.NDArray) else num
    start_is_array = isinstance(start, mx.nd.NDArray)
    stop_is_array = isinstance(stop, mx.nd.NDArray)
    start_shape = []
    if start_is_array:
        start_shape = list(start.shape)
        start = start.reshape((-1,))
    if stop_is_array:
        start_shape = list(stop.shape)
        stop = stop.reshape((-1,))
    if start_is_array and stop_is_array:
        res = [_linspace(strt, stp, num, cont) for strt, stp in zip(start, stop)]
    elif start_is_array and not stop_is_array:
        res = [_linspace(strt, stop, num, cont) for strt in start]
    elif not start_is_array and stop_is_array:
        res = [_linspace(start, stp, num, cont) for stp in stop]
    else:
        return _linspace(start, stop, num, cont)
    new_shape = start_shape + [num]
    res = mx.nd.concat(*res, dim=-1).reshape(new_shape)
    if axis is not None:
        res = mx.nd.swapaxes(res, axis, -1)
    return res

def eye(n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[mx.nd.NDArray] = None,
        device: Optional[str] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    return mx.nd.eye(n_rows, n_cols, k, ctx=cont).astype(dtype)


# Extra #
# ------#

def array(object_in, dtype=None, dev=None):
    cont = _mxnet_init_context(default_device(dev))
    return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))


def logspace(start, stop, num, base=10., axis=None, dev=None):
    power_seq = linspace(start, stop, num, axis, default_device(dev))
    return base ** power_seq

