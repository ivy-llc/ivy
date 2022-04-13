# global
torch_scatter = None
import torch as torch
from typing import List, Tuple, Union, Optional

#local
import ivy

# Array API Standard #
# -------------------#

# noinspection PyShadowingBuiltins
def min(x: torch.Tensor,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False, out: Optional[torch.Tensor]=None)\
        -> torch.Tensor:
    if axis == ():
        if ivy.exists(out):
            ivy.inplace_update(out, x)
        else:
            return x
    if not keepdims and not axis and axis != 0:
        if ivy.exists(out):
            return ivy.inplace_update(out, torch.amin(input=x))
        else:
            return torch.amin(input = x)
    if ivy.exists(out):
        ivy.inplace_update(out, torch.amin(input=x, dim=axis, keepdim=keepdims))
    else:
        return torch.amin(input = x, dim = axis, keepdim = keepdims)


def sum(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        dtype: Optional[torch.dtype] = None,
        keepdims: bool = False)\
        -> torch.Tensor:

    if dtype == None:
        if x.dtype in [torch.int8, torch.int16]:
            dtype = torch.int32
        elif x.dtype == torch.uint8:
            dtype = torch.uint8
        elif x.dtype in [torch.int32, torch.int64]:
            dtype = torch.int64

    if axis is None:
       return torch.sum(input=x,dtype=dtype)
    elif type(axis) == list:
        torch.sum(input = x, dim = axis)
    elif type(axis) == tuple:
        if len(axis) == 0:
            axis = 0
        else:
            return torch.sum(torch.Tensor(
                [torch.sum(input=x, dim=i, dtype=dtype, keepdim=keepdims) for i in axis]), dtype=dtype)

    return torch.sum(input=x, dim=axis, dtype=dtype, keepdim=keepdims)


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


def mean(x: torch.Tensor,
         axis: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdims: bool = False)\
        -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    torch.mean(x, dim=axis, keepdim=keepdims)
    if axis == ():
        return x
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


def std(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> torch.Tensor:
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    if isinstance(axis, int):
        return torch.std(x, dim=axis, keepdim=keepdims)
    dims = len(x.shape)
    axis = tuple([i % dims for i in axis])
    for i, a in enumerate(axis):
        x = torch.std(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x


# Extra #
# ------#

def einsum(equation, *operands):
    return torch.einsum(equation, *operands)
