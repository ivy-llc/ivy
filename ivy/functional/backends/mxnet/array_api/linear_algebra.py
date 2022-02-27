# global
import mxnet as mx
from typing  import Union, Optional, Tuple, Literal


# local
from ivy import inf

def vector_norm(x: mx.ndarray.ndarray.NDArray,
                p: Union[int, float, Literal[inf, - inf]] = 2,
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False)\
                    -> mx.ndarray.ndarray.NDArray:

    return mx.np.linalg.norm(x,p,axis,keepdims)


def cross(x1: mx.ndarray.ndarray.NDArray, x2: mx.ndarray.ndarray.NDArray) -> mx.ndarray.ndarray.NDArray:
    a1 = x1[..., 0:1]
    a2 = x1[..., 1:2]
    a3 = x1[..., 2:3]
    b1 = x2[..., 0:1]
    b2 = x2[..., 1:2]
    b3 = x2[..., 2:3]
    res1 = a2 * b3 - a3 * b2
    res2 = a3 * b1 - a1 * b3
    res3 = a1 * b2 - a2 * b1
    res = mx.nd.concat(res1, res2, res3, dim=-1)
    return res
