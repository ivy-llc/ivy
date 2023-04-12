# global
from typing import (
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
import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array


def moveaxis(
    a: np.ndarray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
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
    out: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not largest:
        indices = np.argsort(x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
    else:
        x = -x
        indices = np.argsort(x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
        x = -x
    topk_res = NamedTuple("top_k", [("values", np.ndarray), ("indices", np.ndarray)])
    val = np.take_along_axis(x, indices, axis=axis)
    return topk_res(val, indices)


def fliplr(
    m: np.ndarray,
    /,
    *,
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


def pad(
    input: np.ndarray,
    pad_width: Union[Sequence[Sequence[int]], np.ndarray, int],
    /,
    *,
    mode: Union[
        Literal[
            "constant",
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
    stat_length: Union[Sequence[Sequence[int]], int] = 1,
    constant_values: Union[Sequence[Sequence[Number]], Number] = 0,
    end_values: Union[Sequence[Sequence[Number]], Number] = 0,
    reflect_type: Literal["even", "odd"] = "even",
    **kwargs: Optional[Any],
) -> np.ndarray:
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
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[np.ndarray]:
    return np.vsplit(ary, indices_or_sections)


def dsplit(
    ary: np.ndarray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[np.ndarray]:
    if ary.ndim < 3:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return np.dsplit(ary, indices_or_sections)


def atleast_1d(*arys: Union[np.ndarray, bool, Number]) -> List[np.ndarray]:
    return np.atleast_1d(*arys)


def dstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.dstack(arrays)


def atleast_2d(*arys: np.ndarray) -> List[np.ndarray]:
    return np.atleast_2d(*arys)


def atleast_3d(*arys: Union[np.ndarray, bool, Number]) -> List[np.ndarray]:
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
    elif mode == "fill" or mode == "drop":
        if "float" in str(arr.dtype):
            fill_value = np.NAN
        elif "uint" in str(arr.dtype):
            fill_value = np.iinfo(arr.dtype).max
        else:
            fill_value = -np.iinfo(arr.dtype).max - 1
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
) -> List[np.ndarray]:
    return np.hsplit(ary, indices_or_sections)


take_along_axis.support_native_out = False


def broadcast_shapes(*shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    return np.broadcast_shapes(*shapes)


broadcast_shapes.support_native_out = False


def expand(
    x: np.ndarray,
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    shape = list(shape)
    for i, dim in enumerate(shape):
        if dim < 0:
            shape[i] = x.shape[i]
    return np.broadcast_to(x, tuple(shape))


expand.support_native_out = False
