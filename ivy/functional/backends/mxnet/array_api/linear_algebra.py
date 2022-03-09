#global
import mxnet.numpy as mx

def det(x:mx.ndarray) \
    -> mx.ndarray:
    return mx.linalg.det(x)
