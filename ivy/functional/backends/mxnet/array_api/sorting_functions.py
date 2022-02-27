import mxnet as mx
from typing import Union, Tuple, Optional, List


def argsort(x: mx.ndarray.ndarray.NDArray,
            axis: int = -1,
            descending: bool = False,
            stable: bool = True)\
            -> mx.ndarray.ndarray.NDArray:
    return mx.nd.array(mx.nd.argsort(mx.nd.array(x), axis=axis, is_ascend=descending))
