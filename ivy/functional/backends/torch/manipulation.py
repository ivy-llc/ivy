# For Review
# global
import ivy
import torch
import math
from numbers import Number
from typing import Union, Optional, Tuple, List, Sequence, Iterable

# noinspection PyProtectedMember
from ivy.functional.ivy.manipulation import _calculate_out_shape


# Array API Standard #
# -------------------#


def concat(
    xs: List[torch.Tensor], /, *, axis: int = 0, out: Optional[torch.Tensor] = None
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
    axis: Union[int, Tuple[int], List[int]] = 0,
) -> torch.Tensor:
    out_shape = _calculate_out_shape(axis, x.shape)
    # torch.reshape since it can operate on contiguous and non_contiguous tensors
    ret = x.reshape(out_shape)
    return ret


expand_dims.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def flip(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
    ret = torch.flip(x, new_axis)
    return ret


flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def permute_dims(
    x: torch.Tensor, /, axes: Tuple[int, ...], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.permute(x, axes)
    return ret


permute_dims.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def reshape(
    x: torch.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
) -> torch.Tensor:
    if copy:
        newarr = torch.clone(x)
        return torch.reshape(newarr, shape)
    return torch.reshape(x, shape)


reshape.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


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

    return torch.roll(x, shift, axis)


roll.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def squeeze(
    x: torch.Tensor,
    /,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
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
            raise ValueError(f"Expected size of axis to be 1 but was {x.shape[axis]}")
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


squeeze.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def stack(
    arrays: Union[Tuple[torch.Tensor], List[torch.Tensor]],
    /,
    *,
    axis: Optional[int] = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ret = torch.stack(arrays, axis, out=out)
    return ret


stack.support_native_out = True


# Extra #
# ------#


def split(
    x,
    /,
    *,
    num_or_size_splits: Optional[Union[int, List[int]]] = None,
    axis: int = 0,
    with_remainder: bool = False,
    out: Optional[torch.Tensor] = None,
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
                    torch.tensor(dim_size) / torch.tensor(num_or_size_splits), out=out
                )
            else:
                num_or_size_splits = tuple(
                    [num_or_size_splits] * num_chunks_int
                    + [int(remainder * num_or_size_splits)]
                )
        else:
            num_or_size_splits = torch.round(
                torch.tensor(dim_size) / torch.tensor(num_or_size_splits), out=out
            )
    elif isinstance(num_or_size_splits, list):
        num_or_size_splits = tuple(num_or_size_splits)
    return list(torch.split(x, num_or_size_splits, axis))


split.support_native_out = True


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
    ret = torch.repeat_interleave(x, repeats, axis)
    return ret


repeat.unsupported_dtypes = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "float16",
)


def tile(
    x: torch.Tensor, /, reps, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if isinstance(reps, torch.Tensor):
        reps = reps.detach().cpu().numpy().tolist()
    ret = x.repeat(reps)
    return ret


tile.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


# noinspection PyUnresolvedReferences
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
    pad_width.reverse()
    pad_width_flat: List[int] = list()
    for pad_width_sec in pad_width:
        for item in pad_width_sec:
            pad_width_flat.append(item)
    ret = torch.nn.functional.pad(x, pad_width_flat, mode="constant", value=value)
    return ret


constant_pad.unsupported_dtypes = ("uint16", "uint32", "uint64")


def zero_pad(
    x: torch.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    out: Optional[torch.Tensor] = None,
):
    return constant_pad(x, pad_width, value=0.0)


zero_pad.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def swapaxes(
    x: torch.Tensor, axis0: int, axis1: int, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = torch.transpose(x, axis0, axis1)
    return ret


swapaxes.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def clip(
    x: torch.Tensor,
    x_min: Union[Number, torch.Tensor],
    x_max: Union[Number, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hasattr(x_min, "dtype"):
        promoted_type = torch.promote_types(x_min.dtype, x_max.dtype)
        promoted_type = torch.promote_types(promoted_type, x.dtype)
        x_min = x_min.to(promoted_type)
        x_max = x_max.to(promoted_type)
        x = x.to(promoted_type)
    ret = torch.clamp(x, x_min, x_max, out=out)
    return ret


clip.support_native_out = True


clip.unsupported_dtypes = ("float16",)
