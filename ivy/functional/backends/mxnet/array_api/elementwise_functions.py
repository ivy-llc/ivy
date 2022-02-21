# global
import mxnet as mx

def bitwise_and(x1: mx.nd.ndarray.NDArray, x2: mx.nd.ndarray.NDArray, /) -> mx.nd.ndarray.NDArray:
    return mx.numpy.bitwise_and(x1, x2)