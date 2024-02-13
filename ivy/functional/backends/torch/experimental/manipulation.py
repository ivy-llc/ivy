# global
from typing import (
    Iterable,
    Optional,
    Union,
    Sequence,
    Tuple,
    NamedTuple,
    List,
    Literal,
    Callable,
    Any,
)
from numbers import Number
from collections import namedtuple
import torch


# local
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    handle_out_argument,
)
from .. import backend_version
import ivy
from ivy.functional.ivy.experimental.manipulation import (
    _to_tf_padding,
    _check_paddle_pad,
    _to_paddle_padding,
)


def moveaxis(
    a: torch.Tensor,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.moveaxis(a, source, destination)


moveaxis.support_native_out = False


def heaviside(
    x1: torch.tensor,
    x2: torch.tensor,
    /,
    *,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.heaviside(
        x1,
        x2,
        out=out,
    )


heaviside.support_native_out = True


@with_supported_dtypes(
    {"2.2 and below": ("float32", "float64", "complex64", "complex128")},
    backend_version,
)
def pad(
    input: torch.Tensor,
    pad_width: Union[Iterable[Tuple[int]], int],
    /,
    *,
    mode: Union[
        Literal[
            "constant",
            "dilated",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
            "empty",
        ],
        Callable,
    ] = "constant",
    stat_length: Union[Iterable[Tuple[int]], int] = 1,
    constant_values: Union[Iterable[Tuple[Number]], Number] = 0,
    end_values: Union[Iterable[Tuple[Number]], Number] = 0,
    reflect_type: Literal["even", "odd"] = "even",
    **kwargs: Optional[Any],
) -> torch.Tensor:
    constant_values = (
        float(constant_values)
        if not isinstance(constant_values, float)
        else constant_values
    )
    pad_width = _to_paddle_padding(pad_width, input.ndim)
    mode = "replicate" if mode == "edge" else "circular" if mode == "wrap" else mode
    if mode == "circular":
        return (
            torch.nn.functional.pad(
                input.unsqueeze(0).unsqueeze(0),
                tuple(pad_width),
                mode=mode,
            )
            .squeeze(0)
            .squeeze(0)
        )
    elif mode == "constant":
        return torch.nn.functional.pad(
            input.unsqueeze(0),
            tuple(pad_width),
            mode=mode,
            value=constant_values,
        ).squeeze(0)
    else:
        return torch.nn.functional.pad(
            input.unsqueeze(0),
            tuple(pad_width),
            mode=mode,
        ).squeeze(0)


pad.partial_mixed_handler = (
    lambda *args, mode="constant", constant_values=0, reflect_type="even", **kwargs: (
        _check_torch_pad(mode, reflect_type, args[1], args[0].shape, constant_values)
    )
)


def _check_torch_pad(mode, reflect_type, pad_width, input_shape, constant_values):
    ndim = len(input_shape)
    pad_width = _to_tf_padding(pad_width, ndim)
    if mode != "constant" and (ndim > 4 or (ndim > 1 and len(pad_width) > ndim - 1)):
        return False
    return _check_paddle_pad(
        mode,
        reflect_type,
        pad_width,
        input_shape,
        constant_values,
        4,
        extend=False,
    ) and (
        mode != "wrap"
        or all(
            pad_width[i][0] <= s and pad_width[i][1] <= s
            for i, s in enumerate(input_shape)
        )
    )


def flipud(
    m: torch.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.flipud(m)


flipud.support_native_out = False


def vstack(
    arrays: Sequence[torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(arrays, tuple):
        arrays = tuple(arrays)
    return torch.vstack(arrays, out=None)


def hstack(
    arrays: Sequence[torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(arrays, tuple):
        arrays = tuple(arrays)
    return torch.hstack(arrays, out=None)


def rot90(
    m: torch.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[int, int] = (0, 1),
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.rot90(m, k, axes)


def top_k(
    x: torch.Tensor,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(k, x.shape[axis])
    topk_res = NamedTuple(
        "top_k", [("values", torch.Tensor), ("indices", torch.Tensor)]
    )
    if not largest:
        indices = torch.argsort(x, dim=axis)
        indices = torch.index_select(indices, axis, torch.arange(k))
    else:
        indices = torch.argsort(-x, dim=axis)
        indices = torch.index_select(indices, axis, torch.arange(k))
    if not sorted:
        indices = torch.sort(indices, dim=axis)[0]
    val = torch.gather(x, axis, indices)
    return topk_res(val, indices)


def fliplr(
    m: torch.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.fliplr(m)


fliplr.support_native_out = False


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, backend_version)
def i0(
    x: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.i0(x, out=out)


i0.support_native_out = True


def flatten(
    x: torch.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    order: Optional[str] = "C",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)


flatten.partial_mixed_handler = (
    lambda *args, copy=None, start_dim=0, end_dim=1, order="C", **kwargs: order == "C"
)


def vsplit(
    ary: torch.Tensor,
    indices_or_sections: Union[int, Sequence[int], torch.Tensor],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[torch.Tensor]:
    if len(ary.shape) < 2:
        raise ivy.utils.exceptions.IvyError(
            "vsplit only works on arrays of 2 or more dimensions"
        )
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=0)


def dsplit(
    ary: torch.Tensor,
    indices_or_sections: Union[int, Sequence[int], torch.Tensor],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[torch.Tensor]:
    if len(ary.shape) < 2:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=2)


def atleast_1d(*arys: torch.Tensor, copy: Optional[bool] = None) -> List[torch.Tensor]:
    transformed = torch.atleast_1d(*arys)
    if isinstance(transformed, tuple):
        return list(transformed)
    return transformed


def dstack(
    arrays: Sequence[torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(arrays, tuple):
        arrays = tuple(arrays)
    return torch.dstack(arrays, out=out)


def atleast_2d(*arys: torch.Tensor, copy: Optional[bool] = None) -> List[torch.Tensor]:
    transformed = torch.atleast_2d(*arys)
    if isinstance(transformed, tuple):
        return list(transformed)
    return transformed


def atleast_3d(
    *arys: Union[torch.Tensor, bool, Number], copy: Optional[bool] = None
) -> List[torch.Tensor]:
    transformed = torch.atleast_3d(*arys)
    if isinstance(transformed, tuple):
        return list(transformed)
    return transformed


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, backend_version)
def take_along_axis(
    arr: torch.Tensor,
    indices: torch.Tensor,
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if arr.ndim != indices.ndim:
        raise ivy.utils.exceptions.IvyException(
            "arr and indices must have the same number of dimensions;"
            + f" got {arr.ndim} vs {indices.ndim}"
        )
    indices = indices.long()
    if mode not in ["clip", "fill", "drop"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes are 'clip', 'fill', 'drop'."
        )
    arr_shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    if mode == "clip":
        max_index = arr.shape[axis] - 1
        indices = torch.clamp(indices, 0, max_index)
    elif mode in {"fill", "drop"}:
        if "float" in str(arr.dtype) or "complex" in str(arr.dtype):
            fill_value = float("nan")
        elif "uint" in str(arr.dtype):
            fill_value = torch.iinfo(arr.dtype).max
        elif "int" in str(arr.dtype):
            fill_value = -torch.iinfo(arr.dtype).max - 1
        indices = torch.where((indices < 0) | (indices >= arr.shape[axis]), -1, indices)
        arr_shape = list(arr_shape)
        arr_shape[axis] = 1
        fill_arr = torch.full(arr_shape, fill_value, dtype=arr.dtype)
        arr = torch.cat([arr, fill_arr], dim=axis)
        indices = torch.where(indices < 0, arr.shape[axis] + indices, indices)
    return torch.take_along_dim(arr, indices, axis, out=out)


def hsplit(
    ary: torch.Tensor,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[torch.Tensor]:
    if len(ary.shape) == 1:
        return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=0)
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=1)


take_along_axis.support_native_out = True


def broadcast_shapes(*shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    return tuple(torch.broadcast_shapes(*shapes))


broadcast_shapes.support_native_out = False


def expand(
    x: torch.Tensor,
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return x.expand(shape)


expand.support_native_out = False


@with_unsupported_dtypes({"2.2 and below": ("complex", "float16")}, backend_version)
def unique_consecutive(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Results = namedtuple(
        "Results",
        ["output", "inverse_indices", "counts"],
    )
    output, inverse_indices, counts = torch.unique_consecutive(
        x,
        return_inverse=True,
        return_counts=True,
        dim=axis,
    )
    return Results(
        output.to(x.dtype),
        inverse_indices,
        counts,
    )


def column_stack(
    arrays: Sequence[torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.column_stack(arrays)


@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, backend_version)
def put_along_axis(
    arr: torch.Tensor,
    indices: torch.Tensor,
    values: Union[int, torch.Tensor],
    axis: int,
    /,
    *,
    mode: Literal["sum", "min", "max", "mul", "replace"] = "replace",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mode_mappings = {
        "sum": "sum",
        "min": "amin",
        "max": "amax",
        "mul": "prod",
        "replace": "replace",
    }
    mode = mode_mappings.get(mode, mode)
    indices = indices.to(torch.int64)
    if mode == "replace":
        return torch.scatter(arr, axis, indices, values, out=out)
    else:
        return torch.scatter_reduce(arr, axis, indices, values, reduce=mode, out=out)


put_along_axis.partial_mixed_handler = lambda *args, mode=None, **kwargs: mode in [
    "replace",
    "sum",
    "mul",
    "mean",
    "max",
    "min",
]


def concat_from_sequence(
    input_sequence: Union[Tuple[torch.Tensor], List[torch.Tensor]],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    is_tuple = type(input_sequence) is tuple
    if is_tuple:
        input_sequence = list(input_sequence)
    if new_axis == 0:
        ret = torch.cat(input_sequence, dim=axis)
        return ret
    elif new_axis == 1:
        ret = torch.stack(input_sequence, dim=axis)
        return ret


def _take_with_axis(
    x: torch.Tensor, indices: torch.Tensor, /, *, axis: int, mode: str
) -> torch.Tensor:
    # has no checks
    # default behaviour is 'raise' like ON CPU
    # additional check is recommended

    x_shape = x.shape[axis]
    if not ivy.exists(axis):
        x = x.flatten()
        x_shape = torch.prod(torch.tensor(x_shape))
    else:
        x_shape = x.shape[axis]

    # wrap
    if mode == "wrap":
        indices = ((indices % x_shape) + x_shape) % x_shape
    # clip
    else:
        indices = torch.clip(indices, 0, x_shape - 1)

    rank = len(x.shape)
    axis = ((axis % rank) + rank) % rank
    slicer = ([slice(None)] * axis) + [indices]
    slicer = tuple(slicer)

    return x[slicer]


def take(
    x: Union[int, List, torch.Tensor],
    indices: Union[int, List, torch.Tensor],
    /,
    *,
    axis: Optional[int] = None,
    mode: str = "clip",
    fill_value: Optional[Number] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mode not in ["raise", "wrap", "clip", "fill"]:
        raise ValueError("mode must be one of 'clip', 'raise', 'wrap', or 'fill'")
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if len(x.shape) == 0:
        x = torch.tensor([x])
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)
    if indices.dtype.is_floating_point:
        indices = indices.to(torch.int64)

    # raise
    if mode == "raise":
        mode = "clip"
        if ivy.exists(axis):
            try:
                x_shape = x.shape[axis]
            except Exception as e:
                rank = len(x.shape)
                raise IndexError(
                    "IndexError: Dimension out of range"
                    f"(expected to be in range of[-{rank}, {rank-1}]"
                    f", but got {axis})"
                ) from e
        else:
            x_shape = torch.prod(torch.tensor(x.shape))

        bound_check = (indices < -x_shape) | (indices >= x_shape)
        if torch.any(torch.tensor(bound_check)):
            raise IndexError("index out of range in self")

    # clip, wrap
    if mode != "fill":
        ret = _take_with_axis(x, indices, axis=axis, mode=mode)
        if ivy.exists(out):
            ivy.inplace_update(out, ret)
        return ret

    # fill
    x_dtype = x.dtype
    if fill_value is None:
        # set according to jax behaviour
        # https://tinyurl.com/66jn68uj
        if x_dtype.is_floating_point or x_dtype.is_complex:
            # NaN for inexact types
            fill_value = float("NaN")
        else:
            if x_dtype == torch.bool:
                # True for booleans
                fill_value = True
            elif str(x_dtype).split(".")[-1].startswith("u"):
                # the largest positive value for unsigned types
                fill_value = torch.iinfo(x_dtype).max
            else:
                # the largest negative value for signed types
                fill_value = torch.iinfo(x_dtype).min

    fill_value = torch.tensor(fill_value, dtype=x_dtype)
    x_shape = x.shape
    ret = _take_with_axis(x, indices, axis=axis, mode="wrap")

    if len(ret.shape) == 0:
        # if scalar (paddle scalar), scalar fill (replace)
        if torch.any(torch.tensor(indices != 0)):
            ret = fill_value
    else:
        if ivy.exists(axis):
            rank = len(x.shape)
            axis = ((axis % rank) + rank) % rank
            x_shape = x_shape[axis]
        else:
            axis = 0
            x_shape = torch.prod(x_shape)

        bound_check = torch.tensor((indices < -x_shape) | (indices >= x_shape))

        if torch.any(bound_check):
            if axis > 0:
                bound_check = torch.broadcast_to(
                    bound_check, (*x.shape[:axis], *bound_check.shape)
                )
            ret[bound_check] = fill_value

    if ivy.exists(out):
        ivy.inplace_update(out, ret)

    return ret


def trim_zeros(a: torch.Tensor, /, *, trim: Optional[str] = "bf") -> torch.Tensor:
    first = 0
    trim = trim.upper()
    if "F" in trim:
        for i in a:
            if i != 0.0:
                break
            else:
                first = first + 1
    last = len(a)
    if "B" in trim:
        for i in torch.flip(a, [0]):
            if i != 0.0:
                break
            else:
                last = last - 1
    return a[first:last]


@handle_out_argument
def unflatten(
    x: torch.Tensor,
    /,
    shape: Tuple[int] = None,
    dim: Optional[int] = 0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    res = torch.unflatten(x, dim, shape)
    return res
