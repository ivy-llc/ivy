# global
from typing import Tuple, Union
import mxnet as mx

# local
from ivy import default_device
from ivy.functional.backends.mxnet import _mxnet_init_context, _1_dim_array_to_flat_array


def zeros(shape: Union[int, Tuple[int, ...]],
          dtype: type = None,
          device: mx.context.Context = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.zeros((1,), ctx=cont).astype(dtype))
    return mx.nd.zeros(shape, ctx=cont).astype(dtype)