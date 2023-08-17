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
    axis: int = 0,
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
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out_shape = _calculate_out_shape(axis, x.shape)
    # torch.reshape since it can operate on contiguous and non_contiguous tensors
    if copy:
        newarr = torch.clone(x)
        return newarr.reshape(out_shape)
    return x.reshape(out_shape)


def flip(
    x: torch.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
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
    if copy:
        newarr = torch.clone(x)
        return torch.flip(newarr, new_axis)
    return torch.flip(x, new_axis)


def permute_dims(
    x: torch.Tensor,
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if copy:
        newarr = torch.clone(x)
        return torch.permute(newarr, axes)
    return torch.permute(x, axes)


def reshape(
    x: torch.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, torch.tensor(shape) != 0, x.shape)
        ]
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
    if torch.is_tensor(axis):
        axis = axis.tolist()
    # manually cover the case when shift is int, and axis is a tuple/list
    if isinstance(shift, int) and (type(axis) in [list, tuple]):
        shift = [shift for _ in range(len(axis))]
    if isinstance(shift, torch.Tensor):
        shift = shift.tolist()
    return torch.roll(x, shift, axis)


def squeeze(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    copy: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(axis, int):
        if x.size(dim=axis) > 1:
            raise ValueError(
                "Expected dimension of size [{}, {}], but found "
                "dimension size {}".format(-x.dim(), x.dim(), axis)
            )
        if x.shape[axis] != 1:
            raise ivy.utils.exceptions.IvyException(
                f"Expected size of axis to be 1 but was {x.shape[axis]}"
            )
        if copy:
            newarr = torch.clone(x)
            return torch.squeeze(newarr, axis)
        return torch.squeeze(x, axis)
    if axis is None:
        if copy:
            newarr = torch.clone(x)
            return torch.squeeze(newarr)
        return torch.squeeze(x)
    newarr = torch.clone(x)
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
            if copy:
                newarr = torch.squeeze(newarr, i)
            else:
                x = torch.squeeze(x, i)
    if copy:
        return newarr
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
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[Union[int, List[int], torch.Tensor]] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> List[torch.Tensor]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise ivy.utils.exceptions.IvyException(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        if copy:
            newarr = torch.clone(x)
            return [newarr]
        return [x]
    dim_size: int = x.shape[axis]
    if num_or_size_splits is None:
        num_or_size_splits = 1
    elif isinstance(num_or_size_splits, torch.Tensor):
        num_or_size_splits = num_or_size_splits.to(torch.int64).tolist()
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
    if copy:
        newarr = torch.clone(x)
        return list(torch.split(newarr, num_or_size_splits, axis))
    return list(torch.split(x, num_or_size_splits, axis))


@with_unsupported_dtypes(
    {"2.0.1 and below": ("int8", "int16", "uint8")}, backend_version
)
def repeat(
    x: torch.Tensor,
    /,
    repeats: Union[int, Iterable[int]],
    *,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if len(x.shape) == 0 and axis in [0, -1]:
        axis = None
    repeats = torch.tensor(repeats)
    return torch.repeat_interleave(x, repeats, axis)


def tile(
    x: torch.Tensor, /, repeats: Sequence[int], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if isinstance(repeats, torch.Tensor):
        repeats = repeats.detach().cpu().numpy().tolist()
    return x.repeat(repeats)


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
    x: torch.Tensor,
    axis0: int,
    axis1: int,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if copy:
        newarr = torch.clone(x)
        return torch.transpose(newarr, axis0, axis1)
    return torch.transpose(x, axis0, axis1)


@with_unsupported_dtypes(
    {"2.0.1 and below": ("bool", "float16", "complex")}, backend_version
)
def clip(
    x: torch.Tensor,
    x_min: Optional[Union[Number, torch.Tensor]] = None,
    x_max: Optional[Union[Number, torch.Tensor]] = None,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if x_min is None and x_max is None:
        raise ValueError("At least one of the x_min or x_max must be provided")

    if x_min is not None and hasattr(x_min, "dtype"):
        x_min = torch.as_tensor(x_min, device=x.device)
    if x_max is not None and hasattr(x_max, "dtype"):
        x_max = torch.as_tensor(x_max, device=x.device)

    if x_min is not None and x_max is not None:
        promoted_type = torch.promote_types(x_min.dtype, x_max.dtype)
        promoted_type = torch.promote_types(promoted_type, x.dtype)
        x_min = x_min.to(promoted_type)
        x_max = x_max.to(promoted_type)
        x = x.to(promoted_type)

    return torch.clamp(x, min=x_min, max=x_max, out=out)


clip.support_native_out = True


def unstack(
    x: torch.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
) -> List[torch.Tensor]:
    if x.shape == ():
        if copy:
            newarr = torch.clone(x)
            return [newarr]
        return [x]
    ret = None
    if copy:
        newarr = torch.clone(x)
        ret = list(torch.unbind(x, axis))
    else:
        ret = list(torch.unbind(x, axis))
    if keepdims:
        return [r.unsqueeze(axis) for r in ret]
    return ret
