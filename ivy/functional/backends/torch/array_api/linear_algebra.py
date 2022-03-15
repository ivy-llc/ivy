<<<<<<< HEAD
<<<<<<< HEAD
#global
import torch

def det(A:torch.Tensor) \
    -> torch.Tensor:
    return torch.linalg.det(A)
=======
=======
>>>>>>> 44b39991812c41d55e3824294efe81336cdd0404
# global
import torch
from typing import Union, Optional, Tuple, Literal
from collections import namedtuple

# local
import ivy as _ivy
from ivy import inf
from collections import namedtuple
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

<<<<<<< HEAD

def det(A:torch.Tensor,
        out:Optional[torch.Tensor]=None) \
    -> torch.Tensor:
    return torch.linalg.det(A,out)
>>>>>>> 3f3cff53faf198659b6bba8b9350017cbbdd0b10
=======
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


def det(x:torch.Tensor) \
    -> torch.Tensor:
    return torch.linalg.det(x)

def slogdet(x:Union[_ivy.Array,_ivy.NativeArray],full_matrices: bool = True) -> Union[_ivy.Array, Tuple[_ivy.Array,...]]:
    results = namedtuple("slogdet", "sign logabsdet")
    sign, logabsdet = torch.linalg.slogdet(x)
    res = results(sign, logabsdet)
    return res

>>>>>>> 44b39991812c41d55e3824294efe81336cdd0404
