# global
import torch
import math
from numbers import Number
from typing import Union, Optional, Tuple, List


def squeeze(x: torch.Tensor,
            axis: Union[int, Tuple[int], List[int]] = None)\
        -> torch.Tensor:
    if isinstance(axis, int):
        if x.shape[axis] > 1:
            raise ValueError('Expected dimension of size 1, but found dimension size {}'.format(x.shape[axis]))
        return torch.squeeze(x, axis)
    if isinstance(axis, tuple):
        axis = list(axis)
    normalise_axis = [(len(x.shape) - abs(element)) if element < 0 else element for element in axis]
    normalise_axis.sort()
    axis_updated_after_squeeze = [ dim - key for (key, dim) in enumerate(normalise_axis)]
    for i in axis_updated_after_squeeze:
        if x.shape[i] > 1:
            raise ValueError('Expected dimension of size 1, but found dimension size {}'.format(x.shape[i]))
        else:
            x = torch.squeeze(x, i)
    return x


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


def expand_dims(x: torch.Tensor,
                axis: Optional[Union[int, Tuple[int], List[int]]] = None) \
        -> torch.Tensor:
    return torch.unsqueeze(x, axis)

  
def permute_dims(x: torch.Tensor,
                axes: Tuple[int,...]) \
        -> torch.Tensor:
    return torch.permute(x, axes)



# Extra #
# ------#


def split(x, num_or_size_splits: Optional[Union[int, List[int]]] = None, axis: int = 0, with_remainder: bool = False)\
        -> List[torch.Tensor]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception('input array had no shape, but num_sections specified was {}'.format(num_or_size_splits))
        return [x]
    dim_size: int = x.shape[axis]
    if num_or_size_splits is None:
        # noinspection PyUnboundLocalVariable
        num_or_size_splits = 1
    elif isinstance(num_or_size_splits, int):
        if with_remainder:
            num_chunks = x.shape[axis] / num_or_size_splits
            num_chunks_int = math.floor(num_chunks)
            remainder = num_chunks - num_chunks_int
            if remainder == 0:
                num_or_size_splits = torch.round(torch.tensor(dim_size) / torch.tensor(num_or_size_splits))
            else:
                num_or_size_splits = tuple([num_or_size_splits] * num_chunks_int + [int(remainder*num_or_size_splits)])
        else:
            num_or_size_splits = torch.round(torch.tensor(dim_size) / torch.tensor(num_or_size_splits))
    elif isinstance(num_or_size_splits, list):
        num_or_size_splits = tuple(num_or_size_splits)
    return list(torch.split(x, num_or_size_splits, axis))


def repeat(x, repeats: Union[int, List[int]], axis: int = None):
    if len(x.shape) == 0 and axis in [0, -1]:
        axis = None
    return torch.repeat_interleave(x, repeats, axis)


def tile(x, reps):
    if isinstance(reps, torch.Tensor):
        reps = reps.detach().cpu().numpy().tolist()
    return x.repeat(reps)




# noinspection PyUnresolvedReferences
def constant_pad(x, pad_width: List[List[int]], value: Number = 0.):
    if x.shape == ():
        x = x.unsqueeze(0)
    if isinstance(pad_width, torch.Tensor):
        pad_width = pad_width.detach().cpu().numpy().tolist()
    pad_width.reverse()
    pad_width_flat: List[int] = list()
    for pad_width_sec in pad_width:
        for item in pad_width_sec:
            pad_width_flat.append(item)
    return torch.nn.functional.pad(x, pad_width_flat, mode='constant', value=value)


def zero_pad(x, pad_width: List[List[int]]):
    return constant_pad(x, pad_width, 0.)


def swapaxes(x, axis0: int, axis1: int):
    return torch.transpose(x, axis0, axis1)