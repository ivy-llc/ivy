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

def squeeze(x: torch.Tensor,
             axis: Union[int, Tuple[int], List[int]])\
             -> torch.Tensor:


    new_axis = axis
    if isinstance(new_axis, int):
        new_axis = [new_axis]
    if isinstance(new_axis, tuple):
        new_axis = sorted(list(new_axis))
    if isinstance(new_axis, list):
        squeezeable_axes = [i for i, ax in enumerate(x.shape) if ax == 1]
        if any(i not in squeezeable_axes for i in new_axis) and not all(i < 0 for i in new_axis):
            raise ValueError
        for ax in new_axis[::-1]:
            if x.shape[ax] > 1:
                raise ValueError
            x = torch.squeeze(x, ax)
        return x

    return torch.squeeze(x, axis)
