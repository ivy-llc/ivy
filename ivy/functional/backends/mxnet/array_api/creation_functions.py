# global
import mxnet as mx
from typing import Tuple, Union, Optional, Iterable

# local
from ivy import default_device
from ivy.functional.backends.mxnet import _mxnet_init_context, _1_dim_array_to_flat_array


def zeros(shape: Union[int, Tuple[int]],
          dtype: Optional[type] = None,
          device: Optional[mx.context.Context] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.zeros((1,), ctx=cont).astype(dtype))
    return mx.nd.zeros(shape, ctx=cont).astype(dtype)


def ones(shape: Union[int, Tuple[int]],
         dtype: Optional[mx.nd.dtype] = None,
         device: Optional[str] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    shape = [shape] if shape is not isinstance(shape, Iterable) else shape
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.ones((1,), ctx=cont).astype(dtype))
    return mx.nd.ones(shape, ctx=cont).astype(dtype)
