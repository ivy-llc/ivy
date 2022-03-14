# global
import torch
from typing import Union, Optional, Tuple, List


# noinspection PyShadowingBuiltins
def flip(x: torch.Tensor,
         axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
         -> torch.Tensor:
    num_dims: int = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        new_axis: List[int] = list(range(num_dims))
    else:
        new_axis: List[int] = axis
    if isinstance(new_axis, int):
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return torch.flip(x, new_axis)

def stack(x: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
          dim: Optional[int] = None)\
          -> torch.Tensor:
    if x is torch.Tensor:
        x = [x]
    if dim is None:
        new_dim = 0
    else:
        new_dim = dim
    return torch.stack(x,dim=new_dim)
