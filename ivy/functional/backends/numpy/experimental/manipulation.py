# global
from typing import (
    Iterable,
    Optional,
    Union,
    Sequence,
    Tuple,
    NamedTuple,
    Literal,
    Callable,
    Any,
    List,
)
from numbers import Number
from collections import namedtuple
import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_supported_dtypes, handle_out_argument

# noinspection PyProtectedMember
from . import backend_version


def moveaxis(
    a: np.ndarray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.moveaxis(a, source, destination)


moveaxis.support_native_out = False


def heaviside(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.heaviside(
        x1,
        x2,
        out=out,
    )


heaviside.support_native_out = True


def flipud(
    m: np.ndarray,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.flipud(m)


flipud.support_native_out = False


def vstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.vstack(arrays)


def hstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.hstack(arrays)


def rot90(
    m: np.ndarray,
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[int, int] = (0, 1),
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.rot90(m, k, axes)


def top_k(
    x: np.ndarray,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, x.shape[axis])
    if not largest:
        indices = np.argsort(x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
    else:
        indices = np.argsort(-x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
    if not sorted:
        indices = np.sort(indices, axis=axis)
    topk_res = NamedTuple("top_k", [("values", np.ndarray), ("indices", np.ndarray)])
    val = np.take_along_axis(x, indices, axis=axis)
    return topk_res(val, indices)


def fliplr(
    m: np.ndarray,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fliplr(m)


fliplr.support_native_out = False


def i0(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.i0(x)


i0.support_native_out = False


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def _slice(operand, start_indices, limit_indices, strides=None):
    strides = [1] * len(operand.shape) if strides is None else strides

    full_slice = ()
    for i, _ in enumerate(operand.shape):
        strides_i = int(strides[i])
        start_i = int(start_indices[i])
        limit_i = int(limit_indices[i])
        full_slice += (slice(start_i, limit_i, strides_i),)
    return operand[full_slice]


def _interior_pad(operand, padding_value, padding_config):
    for axis, (_, _, interior) in enumerate(padding_config):
        if interior > 0:
            new_shape = list(operand.shape)
            new_shape[axis] = new_shape[axis] + (new_shape[axis] - 1) * interior
            new_array = np.full(new_shape, padding_value, dtype=operand.dtype)
            src_indices = np.arange(operand.shape[axis])
            dst_indices = src_indices * (interior + 1)
            index_tuple = [slice(None)] * operand.ndim
            index_tuple[axis] = dst_indices
            new_array[tuple(index_tuple)] = operand
            operand = new_array

    start_indices = [0] * operand.ndim
    limit_indices = [0] * operand.ndim
    for axis, (low, high, _) in enumerate(padding_config):
        if low < 0:
            start_indices[axis] = abs(low)
        if high < 0:
            limit_indices[axis] = high
        else:
            limit_indices[axis] = operand.shape[axis] + 1
    padded = _slice(operand, start_indices, limit_indices)

    pad_width = [(0, 0)] * operand.ndim
    for axis, (low, high, _) in enumerate(padding_config):
        if low > 0 and high > 0:
            pad_width[axis] = (low, high)
        elif low > 0 and not high > 0:
            pad_width[axis] = (low, 0)
        elif high > 0 and not low > 0:
            pad_width[axis] = (0, high)
    padded = np.pad(padded, pad_width, constant_values=padding_value)
    return padded


def pad(
    input: np.ndarray,
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
) -> np.ndarray:
    if mode == "dilated":
        return _interior_pad(input, constant_values, pad_width)
    if callable(mode):
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            **kwargs,
        )
    if mode in ["maximum", "mean", "median", "minimum"]:
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            stat_length=stat_length,
        )
    elif mode == "constant":
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            constant_values=constant_values,
        )
    elif mode == "linear_ramp":
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            end_values=end_values,
        )
    elif mode in ["reflect", "symmetric"]:
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            reflect_type=reflect_type,
        )
    else:
        return np.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
        )


def vsplit(
    ary: np.ndarray,
    indices_or_sections: Union[int, Sequence[int], np.ndarray],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[np.ndarray]:
    if ary.ndim < 2:
        raise ivy.exceptions.IvyError(
            "vsplit only works on arrays of 2 or more dimensions"
        )
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=0)


def dsplit(
    ary: np.ndarray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[np.ndarray]:
    if ary.ndim < 3:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=2)


def atleast_1d(
    *arys: Union[np.ndarray, bool, Number], copy: Optional[bool] = None
) -> List[np.ndarray]:
    return np.atleast_1d(*arys)


def dstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.dstack(arrays)


def atleast_2d(*arys: np.ndarray, copy: Optional[bool] = None) -> List[np.ndarray]:
    return np.atleast_2d(*arys)


def atleast_3d(
    *arys: Union[np.ndarray, bool, Number], copy: Optional[bool] = None
) -> List[np.ndarray]:
    return np.atleast_3d(*arys)


@_scalar_output_to_0d_array
def take_along_axis(
    arr: np.ndarray,
    indices: np.ndarray,
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if arr.ndim != indices.ndim:
        raise ivy.utils.exceptions.IvyException(
            "arr and indices must have the same number of dimensions;"
            + f" got {arr.ndim} vs {indices.ndim}"
        )
    if mode not in ["clip", "fill", "drop"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes are 'clip', 'fill', 'drop'."
        )
    arr_shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    if mode == "clip":
        max_index = arr.shape[axis] - 1
        indices = np.clip(indices, 0, max_index)
    elif mode in ("fill", "drop"):
        if "float" in str(arr.dtype) or "complex" in str(arr.dtype):
            fill_value = np.NAN
        elif "uint" in str(arr.dtype):
            fill_value = np.iinfo(arr.dtype).max
        elif "int" in str(arr.dtype):
            fill_value = -np.iinfo(arr.dtype).max - 1
        else:
            raise TypeError(
                f"Invalid dtype '{arr.dtype}'. Valid dtypes are 'float', 'complex',"
                " 'uint', 'int'."
            )
        indices = np.where((indices < 0) | (indices >= arr.shape[axis]), -1, indices)
        arr_shape = list(arr_shape)
        arr_shape[axis] = 1
        fill_arr = np.full(arr_shape, fill_value, dtype=arr.dtype)
        arr = np.concatenate([arr, fill_arr], axis=axis)
    return np.take_along_axis(arr, indices, axis)


def hsplit(
    ary: np.ndarray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[np.ndarray]:
    if ary.ndim == 1:
        return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=0)
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=1)


take_along_axis.support_native_out = False


def broadcast_shapes(*shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    return np.broadcast_shapes(*shapes)


broadcast_shapes.support_native_out = False


def expand(
    x: np.ndarray,
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    shape = list(shape)
    n_extra_dims = len(shape) - x.ndim
    if n_extra_dims > 0:
        x = np.expand_dims(x, tuple(range(n_extra_dims)))
    for i, dim in enumerate(shape):
        if dim < 0:
            shape[i] = x.shape[i]
    return np.broadcast_to(x, tuple(shape))


expand.support_native_out = False


def concat_from_sequence(
    input_sequence: Union[Tuple[np.ndarray], List[np.ndarray]],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    is_tuple = type(input_sequence) is tuple
    if is_tuple:
        input_sequence = list(input_sequence)
    if new_axis == 0:
        ret = np.concatenate(input_sequence, axis=axis)
        return ret
    elif new_axis == 1:
        ret = np.stack(input_sequence, axis=axis)
        return ret


def unique_consecutive(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Results = namedtuple(
        "Results",
        ["output", "inverse_indices", "counts"],
    )
    x_shape = None
    if axis is None:
        x_shape = x.shape
        x = x.flatten()
        axis = -1
    if axis < 0:
        axis += x.ndim
    sub_arrays = np.split(
        x,
        np.where(
            np.any(
                np.diff(x, axis=axis) != 0,
                axis=tuple(i for i in np.arange(x.ndim) if i != axis),
            )
        )[0]
        + 1,
        axis=axis,
    )
    output = np.concatenate(
        [np.unique(sub_array, axis=axis) for sub_array in sub_arrays],
        axis=axis,
    )
    counts = np.array([sub_array.shape[axis] for sub_array in sub_arrays])
    inverse_indices = np.repeat(np.arange(len(counts)), counts)
    if x_shape:
        inverse_indices = np.reshape(inverse_indices, x_shape)
    return Results(
        output.astype(x.dtype),
        inverse_indices,
        counts,
    )


def fill_diagonal(
    a: np.ndarray,
    v: Union[int, float, np.ndarray],
    /,
    *,
    wrap: bool = False,
) -> np.ndarray:
    np.fill_diagonal(a, v, wrap=wrap)
    return a


@_scalar_output_to_0d_array
def take(
    x: Union[int, List, np.ndarray],
    indices: Union[int, List, np.ndarray],
    /,
    *,
    axis: Optional[int] = None,
    mode: str = "raise",
    fill_value: Optional[Number] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mode not in ["raise", "wrap", "clip", "fill"]:
        raise ValueError("mode must be one of 'clip', 'raise', 'wrap', or 'fill'")

    # raise, clip, wrap
    if mode != "fill":
        return np.take(x, indices, axis=axis, mode=mode, out=out)

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if len(x.shape) == 0:
        x = np.array([x])
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if np.issubdtype(indices.dtype, np.floating):
        indices = indices.astype(np.int64)

    # fill
    x_dtype = x.dtype
    if fill_value is None:
        # set according to jax behaviour
        # https://tinyurl.com/66jn68uj
        # NaN for inexact types (let fill_value as None)
        if not np.issubdtype(x_dtype, np.inexact):
            if np.issubdtype(x_dtype, np.bool_):
                # True for booleans
                fill_value = True
            elif np.issubdtype(x_dtype, np.unsignedinteger):
                # the largest positive value for unsigned types
                fill_value = np.iinfo(x_dtype).max
            else:
                # the largest negative value for signed types
                fill_value = np.iinfo(x_dtype).min

    fill_value = np.array(fill_value, dtype=x_dtype)
    x_shape = x.shape
    ret = np.take(x, indices, axis=axis, mode="wrap")

    if len(ret.shape) == 0:
        # if scalar, scalar fill (replace)
        if np.any(indices != 0):
            ret = fill_value
    else:
        if ivy.exists(axis):
            rank = len(x.shape)
            axis = ((axis % rank) + rank) % rank
            x_shape = x_shape[axis]
        else:
            axis = 0
            x_shape = np.prod(x_shape)

        bound_check = (indices < -x_shape) | (indices >= x_shape)

        if np.any(bound_check):
            if axis > 0:
                bound_check = np.broadcast_to(
                    bound_check, (*x.shape[:axis], *bound_check.shape)
                )
            ret[bound_check] = fill_value

    if ivy.exists(out):
        ivy.inplace_update(out, ret)

    return ret


take.support_native_out = True


def trim_zeros(
    a: np.ndarray,
    /,
    *,
    trim: Optional[str] = "fb",
) -> np.ndarray:
    return np.trim_zeros(a, trim=trim)


def column_stack(
    arrays: Sequence[np.ndarray], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.column_stack(arrays)


@with_supported_dtypes(
    {"1.25.2 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def put_along_axis(
    arr: np.ndarray,
    indices: np.ndarray,
    values: Union[int, np.ndarray],
    axis: int,
    /,
    *,
    mode: Literal["sum", "min", "max", "mul", "replace"] = "replace",
    out: Optional[np.ndarray] = None,
):
    ret = arr.copy()
    values = np.asarray(values)
    np.put_along_axis(ret, indices, values, axis)
    return ivy.inplace_update(out, ret) if ivy.exists(out) else ret


put_along_axis.partial_mixed_handler = lambda *args, mode=None, **kwargs: mode in [
    "replace",
]


@handle_out_argument
def unflatten(
    x: np.ndarray,
    /,
    shape: Tuple[int] = None,
    dim: Optional[int] = 0,
    *,
    out: Optional[np.ndarray] = None,
    order: Optional[str] = None,
) -> np.ndarray:
    dim = abs(len(x.shape) + dim) if dim < 0 else dim
    res_shape = x.shape[:dim] + shape + x.shape[dim + 1 :]
    res = np.reshape(x, res_shape)
    return res
