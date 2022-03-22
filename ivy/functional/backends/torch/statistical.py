# global
torch_scatter = None
import torch as torch
from typing import List, Tuple, Union, Optional


# Array API Standard #
# -------------------#

# noinspection PyShadowingBuiltins
def min(x: torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis == ():
        return x
    if not keepdims and not axis and axis != 0:
        return torch.amin(input = x)
    return torch.amin(input = x, dim = axis, keepdim = keepdims)


def sum(x: torch.Tensor,
        axis: Optional[List[int]] = None,
        keepdims: bool = False)\
        -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    return torch.sum(x, dim=axis, keepdim=keepdims)


def prod(x: torch.Tensor,
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[torch.dtype] = None,
         keepdims: bool = False)\
        -> torch.Tensor:

    if dtype is None:
        if x.dtype in [torch.int8, torch.int16] :
            dtype = torch.int32
        elif x.dtype == torch.uint8:
            dtype = torch.uint8
        elif x.dtype in [torch.int64, torch.int32]:
            dtype = torch.int64

    if axis is None:
        axis = x.dim()-1
    elif type(axis) == tuple:
        if len(axis) == 0:
            axis = x.dim()-1
        else:
            return torch.prod(torch.Tensor(
                [torch.prod(input=x, dim=i, dtype=dtype, keepdim=keepdims) for i in axis]), dtype=dtype)

    return torch.prod(input=x, dim=axis, dtype=dtype, keepdim=keepdims)


def mean(x, axis: Optional[List[int]] = None, keepdims: bool = False):
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    return torch.mean(x, dim=axis, keepdim=keepdims)


# noinspection PyShadowingBuiltins
def max(x: torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis == (): return x
    if not keepdims and not axis and axis !=0: return torch.amax(input = x)
    return torch.amax(input = x, dim = axis, keepdim = keepdims)


def var(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    if isinstance(axis, int):
        return torch.var(x, dim=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = tuple([i % dims for i in axis])
    for i, a in enumerate(axis):
        x = torch.var(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x


# Extra #
# ------#

def einsum(equation, *operands):
    return torch.einsum(equation, *operands)
