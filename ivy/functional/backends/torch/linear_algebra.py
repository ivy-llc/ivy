# global
import torch
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

# local
import ivy as _ivy
from ivy import inf
from collections import namedtuple
import ivy as _ivy


def inv(x):
    return torch.inverse(x)


def matrix_transpose(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.swapaxes(x, -1, -2)


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


def diagonal(x: torch.Tensor,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> torch.Tensor:
    return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


def svdvals(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(x)


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

        
def matmul(x1: torch.Tensor,
           x2: torch.Tensor) -> torch.Tensor:
    dtype_from = torch.promote_types(x1.dtype, x2.dtype)
    x1 = x1.type(dtype_from)
    x2 = x2.type(dtype_from)
    ret = torch.matmul(x1, x2)
    return ret.type(dtype_from)


def slogdet(x:Union[_ivy.Array,_ivy.NativeArray],full_matrices: bool = True) -> Union[_ivy.Array, Tuple[_ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = torch.linalg.slogdet(x)
    res = results(sign, logabsdet)
    return res


def trace(x: torch.Tensor,
          offset: int = 0)\
              -> torch.Tensor:
    return torch.trace(x, offset)
