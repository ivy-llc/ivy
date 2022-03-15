<<<<<<< HEAD
#global
import torch

def det(A:torch.Tensor) \
    -> torch.Tensor:
    return torch.linalg.det(A)
=======
# global
import torch
from typing import Union, Optional, Tuple, Literal

# local
from ivy import inf

def vector_norm(x: torch.Tensor,
                p: Union[int, float, Literal[inf, - inf]] = 2,
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False)\
        -> torch.Tensor:

    py_normalized_vector = torch.linalg.vector_norm(x, p, axis, keepdims)

    if py_normalized_vector.shape == ():
        return torch.unsqueeze(py_normalized_vector, 0)

    return py_normalized_vector


def det(A:torch.Tensor,
        out:Optional[torch.Tensor]=None) \
    -> torch.Tensor:
    return torch.linalg.det(A,out)
>>>>>>> 3f3cff53faf198659b6bba8b9350017cbbdd0b10
