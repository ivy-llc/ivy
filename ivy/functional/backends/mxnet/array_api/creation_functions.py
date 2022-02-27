# global
import mxnet as mx
from typing import Tuple, Union, Optional, Iterable

# local
from ivy import default_device,dtype_from_str,default_dtype
from ivy.functional.backends.mxnet import _mxnet_init_context


def empty(shape: Union[int, Tuple[int]],
          dtype: Optional[type] = None,
          device: Optional[mx.context.Context] = None) \
        -> mx.ndarray.ndarray.NDArray:
    cont = _mxnet_init_context(default_device(device))
    return mx.nd.empty(shape, dtype_from_str(default_dtype(dtype)), cont)
