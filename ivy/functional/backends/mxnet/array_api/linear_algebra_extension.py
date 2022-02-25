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
