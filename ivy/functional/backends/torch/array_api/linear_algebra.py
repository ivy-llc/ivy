# global
import torch
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

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


def diagonal(x: torch.Tensor,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> torch.Tensor:
    return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


def qr(x: torch.Tensor,
       mode: str = 'reduced') -> namedtuple('qr', ['Q', 'R']):
    res = namedtuple('qr', ['Q', 'R'])
    if mode == 'reduced':
        q, r = torch.qr(x, some=True)
        return res(q, r)
    elif mode == 'complete':
        q, r = torch.qr(x, some=False)
        return res(q, r)
    else:
        raise Exception("Only 'reduced' and 'complete' qr modes are allowed for the torch backend.")
