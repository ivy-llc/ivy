 # global
import mxnet as mx
import numpy as _np
from typing  import Union, Optional, Tuple, Literal


# local
from ivy import inf
import ivy as _ivy
 
def vector_norm(x: mx.ndarray.ndarray.NDArray,
                p: Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False)\
                    -> mx.ndarray.ndarray.NDArray:
                
    return mx.np.linalg.norm(x,p,axis,keepdims)

# noinspection PyPep8Naming
def svd(x:Union[_ivy.Array,_ivy.NativeArray],full_matrices: bool = True) -> Union[_ivy.Array, Tuple[_ivy.Array,...]]:
    U, D, VT=_np.linalg.svd(x, full_matrices=full_matrices)
    return U, D, VT