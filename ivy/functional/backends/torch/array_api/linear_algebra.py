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


def cross(x1: torch.Tensor, x2: torch.Tensor, /, *, axis: Optional[int] = -1) -> torch.Tensor:
    datatype = torch.promote_types(x1.dtype, x2.dtype)
    if x1.dtype != x2.dtype:
        x1 = torch.tensor(x1, dtype=datatype)
        x2 = torch.tensor(x1, dtype=datatype)
    return torch.cross(x1, x2, dim=axis)
