import mxnet as mx
from typing import Union, Tuple, Optional, List

from ivy.functional.backends.mxnet import reduce_prod, _flat_array_to_1_dim_array, _1_dim_array_to_flat_array


def var(x: mx.ndarray.ndarray.NDArray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) -> mx.ndarray.ndarray.NDArray:
    return mx.nd.array(mx.nd.array(x).asnumpy().var(axis=axis, keepdims=keepdims))
