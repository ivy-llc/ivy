# global
import torch
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

# local
from ivy import inf
import ivy as _ivy

def vector_norm(x: torch.Tensor,
                p: Union[int, float, Literal[inf, - inf]] = 2,
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False)\
        -> torch.Tensor:

    py_normalized_vector = torch.linalg.vector_norm(x, p, axis, keepdims)

    if py_normalized_vector.shape == ():
        return torch.unsqueeze(py_normalized_vector, 0)

    return py_normalized_vector

# noinspection PyPep8Naming
def svd(x:torch.Tensor,full_matrices: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor,...]]:
    results=namedtuple("svd", "U S Vh")

    U, D, VT = torch.linalg.svd(x, full_matrices=full_matrices)
    res=results(U, D, VT)
    return res
