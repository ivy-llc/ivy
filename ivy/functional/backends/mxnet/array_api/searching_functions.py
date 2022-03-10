import mxnet as mx
from typing import Optional, Union


def argmax(
    x:mx.ndarray.ndarray.NDArray,
    axis:Optional[int] = None, 
    out : Optional[mx.ndarray.ndarray.NDArray] = None,
    keepdims: bool = False
    ) -> mx.ndarray.ndarray.NDArray:
    return mx.nd.argmax(x,axis=axis,out=out, keepdims=keepdims)