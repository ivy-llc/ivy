# global
import mxnet as mx
from collections.abc import Iterable
from typing import Union, Optional, Tuple


# local
from ivy.functional.ivy.core import default_device
from ivy.functional.backends.mxnet import _1_dim_array_to_flat_array, _mxnet_init_context

# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[mx.nd.dtype] = 'float32',
         device: Optional[str] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    shape = [shape] if shape is not isinstance(shape, Iterable) else shape
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.ones((1,), ctx=cont).astype(dtype))
    return mx.nd.ones(shape, ctx=cont).astype(dtype)