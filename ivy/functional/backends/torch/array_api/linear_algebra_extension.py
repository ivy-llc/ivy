# global
from typing  import Union, Optional, Tuple, Literal
import torch as torch_

# local
inf = float("inf")

def vector_norm(x: torch_.Tensor, 
                p:Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int, ...]]] = None, 
                keepdims: bool = False)\
                 -> torch_.Tensor :


    py_normalized_vector_ = torch_.linalg.vector_norm(x,p,axis,keepdims)

    if py_normalized_vector_.shape == tuple():
        return torch_.unsqueeze(py_normalized_vector_, 0)
    return py_normalized_vector_