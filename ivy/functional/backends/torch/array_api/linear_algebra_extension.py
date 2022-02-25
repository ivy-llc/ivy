# global
from typing  import Union, Optional, Tuple, Literal
import torch as torch_

# local
from ivy import inf

def vector_norm(x: torch_.Tensor, 
                p: Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False)\
                 -> torch_.Tensor :

    py_normalized_vector = torch_.linalg.vector_norm(x,p,axis,keepdims)

    if py_normalized_vector.shape == tuple():
        return torch_.unsqueeze(py_normalized_vector, 0)

    return py_normalized_vector