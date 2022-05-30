# global
import torch
import math
from numbers import Number
from typing import Union, Optional, Tuple, List

import ivy


def roll(
    x: torch.Tensor,
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> torch.Tensor:

    # manually cover the case when shift is int, and axis is a tuple/list
    if isinstance(shift, int) and (type(axis) in [list, tuple]):
        shift = [shift for _ in range(len(axis))]

    return torch.roll(x, shift, axis)


def squeeze(
    x: torch.Tensor,
    axis: Union[int, Tuple[int], List[int]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(axis, int):
        if x.shape[axis] > 1:
            raise ValueError(
                "Expected dimension of size 1, but found dimension size {}".format(
                    x.shape[axis]
                )
            )
        ret = torch.squeeze(x, axis)
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret
    elif isinstance(axis, tuple):
        axis = list(axis)
    normalise_axis = [
        (len(x.shape) - abs(element)) if element < 0 else element for element in axis
    ]
    normalise_axis.sort()
    axis_updated_after_squeeze = [dim - key for (key, dim) in enumerate(normalise_axis)]
    for i in axis_updated_after_squeeze:
        if x.shape[i] > 1:
            raise ValueError(
                "Expected dimension of size 1, but found dimension size {}".format(
                    x.shape[i]
                )
            )
        else:
            x = torch.squeeze(x, i)
    if ivy.exists(out):
        return ivy.inplace_update(out, x)
    return x


# noinspection PyShadowingBuiltins
def flip(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_dims: int = len(x.shape)
    if not num_dims:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
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
    ret = torch.flip(x, new_axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def expand_dims(
    x: torch.Tensor, axis: int = 0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.unsqueeze(x, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def permute_dims(
    x: torch.Tensor, axes: Tuple[int, ...], out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.permute(x, axes)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def stack(
    x: Union[Tuple[torch.Tensor], List[torch.Tensor]],
    axis: Optional[int] = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ret = torch.stack(x, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def reshape(
    x: torch.Tensor,
    shape: Tuple[int, ...],
    copy: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ret = torch.reshape(x, shape)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def concat(
    xs: List[torch.Tensor], axis: int = 0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if axis is None:
        is_tuple = type(xs) is tuple
        if is_tuple:
            xs = list(xs)
        for i in range(len(xs)):
            xs[i] = torch.flatten(xs[i])
        if is_tuple:
            xs = tuple(xs)
        axis = 0
    return torch.cat(xs, dim=axis, out=out)


# Extra #
# ------#


def split(
    x,
    num_or_size_splits: Optional[Union[int, List[int]]] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> List[torch.Tensor]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
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
                num_or_size_splits = torch.round(
                    torch.tensor(dim_size) / torch.tensor(num_or_size_splits)
                )
            else:
                num_or_size_splits = tuple(
                    [num_or_size_splits] * num_chunks_int
                    + [int(remainder * num_or_size_splits)]
                )
        else:
            num_or_size_splits = torch.round(
                torch.tensor(dim_size) / torch.tensor(num_or_size_splits)
            )
    elif isinstance(num_or_size_splits, list):
        num_or_size_splits = tuple(num_or_size_splits)
    return list(torch.split(x, num_or_size_splits, axis))


def repeat(
    x: torch.Tensor,
    repeats: Union[int, List[int]],
    axis: int = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(x.shape) == 0 and axis in [0, -1]:
        axis = None
    ret = torch.repeat_interleave(x, repeats, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tile(x: torch.Tensor, reps, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(reps, torch.Tensor):
        reps = reps.detach().cpu().numpy().tolist()
    ret = x.repeat(reps)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# noinspection PyUnresolvedReferences
def constant_pad(
    x: torch.Tensor,
    pad_width: List[List[int]],
    value: Number = 0.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.shape == ():
        x = x.unsqueeze(0)
    if isinstance(pad_width, torch.Tensor):
        pad_width = pad_width.detach().cpu().numpy().tolist()
    pad_width.reverse()
    pad_width_flat: List[int] = list()
    for pad_width_sec in pad_width:
        for item in pad_width_sec:
            pad_width_flat.append(item)
    ret = torch.nn.functional.pad(x, pad_width_flat, mode="constant", value=value)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def zero_pad(
    x: torch.Tensor, pad_width: List[List[int]], out: Optional[torch.Tensor] = None
):
    return constant_pad(x, pad_width, 0.0, out=out)


def swapaxes(
    x: torch.Tensor, axis0: int, axis1: int, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.transpose(x, axis0, axis1)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def clip(
    x: torch.Tensor,
    x_min: Union[Number, torch.Tensor],
    x_max: Union[Number, torch.Tensor],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hasattr(x_min, "dtype"):
        promoted_type = torch.promote_types(x_min.dtype, x_max.dtype)
        promoted_type = torch.promote_types(promoted_type, x.dtype)
        x_min = x_min.to(promoted_type)
        x_max = x_max.to(promoted_type)
        x = x.to(promoted_type)
    ret = torch.clamp(x, x_min, x_max, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret
