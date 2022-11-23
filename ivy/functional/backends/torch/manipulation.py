# global
import math
from numbers import Number
from typing import Union, Optional, Tuple, List, Sequence, Iterable

import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes

# noinspection PyProtectedMember
from ivy.functional.ivy.manipulation import _calculate_out_shape
from . import backend_version


def _reshape_fortran_torch(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(shape[::-1]).permute(list(range(len(shape)))[::-1])


# Array API Standard #
# -------------------#


def concat(
    xs: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    /,
    *,
    axis: Optional[int] = 0,
    out: Optional[torch.Tensor] = None,
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


concat.support_native_out = True


def expand_dims(
    x: torch.Tensor,
    /,
    *,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out_shape = _calculate_out_shape(axis, x.shape)
    # torch.reshape since it can operate on contiguous and non_contiguous tensors
    return x.reshape(out_shape)


def flip(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if isinstance(new_axis, int):
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    return torch.flip(x, new_axis)


def permute_dims(
    x: torch.Tensor,
    /,
    axes: Tuple[int, ...],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.permute(x, axes)


def reshape(
    x: torch.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: Optional[str] = "C",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ivy.assertions.check_elem_in_list(order, ["C", "F"])
    if copy:
        newarr = torch.clone(x)
        if order == "F":
            return _reshape_fortran_torch(newarr, shape)
        return torch.reshape(newarr, shape)
    if order == "F":
        return _reshape_fortran_torch(x, shape)
    return torch.reshape(x, shape)


def roll(
    x: torch.Tensor,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # manually cover the case when shift is int, and axis is a tuple/list
    if isinstance(shift, int) and (type(axis) in [list, tuple]):
        shift = [shift for _ in range(len(axis))]
    if isinstance(shift, torch.Tensor):
        shift = shift.tolist()
        shift = tuple([shift])
    return torch.roll(x, shift, axis)


def squeeze(
    x: torch.Tensor,
    /,
    axis: Union[int, Sequence[int]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(axis, int):
        if x.size(dim=axis) > 1:
            raise ValueError(
                "Expected dimension of size [{}, {}], but found "
                "dimension size {}".format(-x.dim(), x.dim(), axis)
            )
        if x.shape[axis] != 1:
            raise ivy.exceptions.IvyException(
                f"Expected size of axis to be 1 but was {x.shape[axis]}"
            )
        return torch.squeeze(x, axis)
    if axis is None:
        return torch.squeeze(x)
    if isinstance(axis, tuple):
        axis = list(axis)
    normalise_axis = [
        (len(x.shape) - abs(element)) if element < 0 else element for element in axis
    ]
    normalise_axis.sort()
    axis_updated_after_squeeze = [dim - key for (key, dim) in enumerate(normalise_axis)]
    dim = x.dim()
    for i in axis_updated_after_squeeze:
        shape = x.shape[i]
        if shape > 1 and (shape < -dim or dim <= shape):
            raise ValueError(
                "Expected dimension of size [{}, {}], "
                "but found dimension size {}".format(-dim, dim, shape)
            )
        else:
            x = torch.squeeze(x, i)
    return x


def stack(
    arrays: Union[Tuple[torch.Tensor], List[torch.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.stack(arrays, axis, out=out)


stack.support_native_out = True


# Extra #
# ------#


def split(
    x: torch.Tensor,
    /,
    *,
    num_or_size_splits: Optional[Union[int, List[int]]] = None,
    axis: Optional[int] = 0,
    with_remainder: Optional[bool] = False,
) -> List[torch.Tensor]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise ivy.exceptions.IvyException(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        return [x]
    dim_size: int = x.shape[axis]
    if num_or_size_splits is None:
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


@with_unsupported_dtypes({"1.11.0": ("int8", "int16", "uint8")}, backend_version)
def repeat(
    x: torch.Tensor,
    /,
    repeats: Union[int, Iterable[int]],
    *,
    axis: int = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(x.shape) == 0 and axis in [0, -1]:
        axis = None
    repeats = torch.tensor(repeats)
    return torch.repeat_interleave(x, repeats, axis)


def tile(
    x: torch.Tensor, /, reps: Sequence[int], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if isinstance(reps, torch.Tensor):
        reps = reps.detach().cpu().numpy().tolist()
    return x.repeat(reps)


def constant_pad(
    x: torch.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.shape == ():
        x = x.unsqueeze(0)
    if isinstance(pad_width, torch.Tensor):
        pad_width = pad_width.detach().cpu().numpy().tolist()
    pad_width_flat: List[int] = list()
    for pad_width_sec in reversed(pad_width):
        for item in pad_width_sec:
            pad_width_flat.append(item)
    return torch.nn.functional.pad(x, pad_width_flat, mode="constant", value=value)


def zero_pad(
    x: torch.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    out: Optional[torch.Tensor] = None,
):
    return constant_pad(x, pad_width, value=0.0)


def swapaxes(
    x: torch.Tensor, axis0: int, axis1: int, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.transpose(x, axis0, axis1)


@with_unsupported_dtypes({"1.11.0": ("float16",)}, backend_version)
def clip(
    x: torch.Tensor,
    x_min: Union[Number, torch.Tensor],
    x_max: Union[Number, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ivy.assertions.check_less(x_min, x_max, message="min values must be less than max")
    if hasattr(x_min, "dtype"):
        promoted_type = torch.promote_types(x_min.dtype, x_max.dtype)
        promoted_type = torch.promote_types(promoted_type, x.dtype)
        x_min = x_min.to(promoted_type)
        x_max = x_max.to(promoted_type)
        x = x.to(promoted_type)
    return torch.clamp(x, x_min, x_max, out=out)


clip.support_native_out = True


def unstack(
    x: torch.Tensor, /, *, axis: int = 0, keepdims: bool = False
) -> List[torch.Tensor]:
    if x.shape == ():
        return [x]
    ret = list(torch.unbind(x, axis))
    if keepdims:
        return [r.unsqueeze(axis) for r in ret]
    return ret
