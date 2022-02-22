 # global
import mxnet as mx
from typing  import Union, Optional, Tuple, Literal

# local
inf = float("inf")
MxArray = mx.ndarray.ndarray.NDArray
 
def vector_norm(x: MxArray,
                p:Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int, ...]]] = None, 
                keepdims: bool = False)\ 
                -> MxArray:
    return mx.np.linalg.norm(x,p,axis,keepdims)