# local
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
import jax.numpy as jnp
import jax.lax as jlax
from numbers import Number
from collections import namedtuple

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def moveaxis(
    a: JaxArray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.moveaxis(a, source, destination)


def heaviside(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.heaviside(x1, x2)


def flipud(
    m: JaxArray,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.flipud(m)


def vstack(
    arrays: Sequence[JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.vstack(arrays)


def hstack(
    arrays: Sequence[JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.hstack(arrays)


def rot90(
    m: JaxArray,
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[int, int] = (0, 1),
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if isinstance(axes, list):
        axes = tuple(axes)
    return jnp.rot90(m, k, axes)


def top_k(
    x: JaxArray,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[Tuple[JaxArray, JaxArray]] = None,
) -> Tuple[JaxArray, JaxArray]:
    k = min(k, x.shape[axis])
    if not largest:
        indices = jnp.argsort(x, axis=axis)
        indices = jnp.take(indices, jnp.arange(k), axis=axis)
    else:
        indices = jnp.argsort(-x, axis=axis)
        indices = jnp.take(indices, jnp.arange(k), axis=axis)
    if not sorted:
        indices = jnp.sort(indices, axis=axis)
    topk_res = NamedTuple("top_k", [("values", JaxArray), ("indices", JaxArray)])
    val = jnp.take_along_axis(x, indices, axis=axis)
    return topk_res(val, indices)


def fliplr(
    m: JaxArray,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fliplr(m)


def i0(
    x: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.i0(x)


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def _to_nested_tuple(nested_list):
    ret = ()
    if hasattr(nested_list, "__iter__"):
        for inner_list in nested_list:
            if hasattr(inner_list, "__iter__"):
                ret += (tuple(inner_list),)
            else:
                ret += (inner_list,)
        return ret
    if ret == ():
        return nested_list


def pad(
    input: JaxArray,
    pad_width: Union[Sequence[Sequence[int]], JaxArray, int],
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
    stat_length: Union[Sequence[Sequence[int]], int] = 1,
    constant_values: Union[Sequence[Sequence[Number]], Number] = 0,
    end_values: Union[Sequence[Sequence[Number]], Number] = 0,
    reflect_type: Literal["even", "odd"] = "even",
    **kwargs: Optional[Any],
) -> JaxArray:
    pad_width = _to_nested_tuple(pad_width)
    stat_length = _to_nested_tuple(stat_length)
    constant_values = _to_nested_tuple(constant_values)
    end_values = _to_nested_tuple(end_values)
    input_dtype = input.dtype

    if mode == "dilated":
        if ivy.as_ivy_dtype(type(constant_values)) != input_dtype:
            padding_value = ivy.native_array(constant_values, dtype=input_dtype)
        else:
            padding_value = constant_values
        padded = jlax.pad(input, padding_value, pad_width)
        return padded

    if callable(mode):
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            **kwargs,
        )
    elif mode in ["maximum", "mean", "median", "minimum"]:
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            stat_length=stat_length,
        )
    elif mode == "constant":
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            constant_values=constant_values,
        )
    elif mode == "linear_ramp":
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            end_values=end_values,
        )
    elif mode in ["reflect", "symmetric"]:
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            reflect_type=reflect_type,
        )
    else:
        ret = jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
        )
    if jnp.issubdtype(input_dtype, jnp.integer) and mode in ["mean", "median"]:
        ret = jnp.round(ret).astype(input_dtype)
    return ret


def vsplit(
    ary: JaxArray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[JaxArray]:
    return jnp.vsplit(ary, indices_or_sections)


def dsplit(
    ary: JaxArray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[JaxArray]:
    if len(ary.shape) < 3:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return jnp.dsplit(ary, indices_or_sections)


def atleast_1d(
    *arys: Union[JaxArray, bool, Number], copy: Optional[bool] = None
) -> List[JaxArray]:
    return jnp.atleast_1d(*arys)


def dstack(
    arrays: Sequence[JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.dstack(arrays)


def atleast_2d(*arys: JaxArray, copy: Optional[bool] = None) -> List[JaxArray]:
    return jnp.atleast_2d(*arys)


def atleast_3d(
    *arys: Union[JaxArray, bool, Number], copy: Optional[bool] = None
) -> List[JaxArray]:
    return jnp.atleast_3d(*arys)


def take_along_axis(
    arr: JaxArray,
    indices: JaxArray,
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if arr.ndim != indices.ndim and axis is not None:
        raise ivy.utils.exceptions.IvyException(
            "arr and indices must have the same number of dimensions;"
            + f" got {arr.ndim} vs {indices.ndim}"
        )
    return jnp.take_along_axis(arr, indices, axis, mode=mode)


def hsplit(
    ary: JaxArray,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[JaxArray]:
    return jnp.hsplit(ary, indices_or_sections)


def broadcast_shapes(*shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    return jnp.broadcast_shapes(*shapes)


def expand(
    x: JaxArray,
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    shape = list(shape)
    for i, dim in enumerate(shape):
        if dim < 0:
            shape[i] = x.shape[i]
    return jnp.broadcast_to(x, tuple(shape))


def concat_from_sequence(
    input_sequence: Union[Tuple[JaxArray], List[JaxArray]],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    is_tuple = type(input_sequence) is tuple
    if is_tuple:
        input_sequence = list(input_sequence)
    if new_axis == 0:
        ret = jnp.concatenate(input_sequence, axis=axis)
        return ret
    elif new_axis == 1:
        ret = jnp.stack(input_sequence, axis=axis)
        return ret


def unique_consecutive(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[JaxArray, JaxArray, JaxArray]:
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
    sub_arrays = jnp.split(
        x,
        jnp.where(
            jnp.any(
                jnp.diff(x, axis=axis) != 0,
                axis=tuple(i for i in jnp.arange(x.ndim) if i != axis),
            )
        )[0]
        + 1,
        axis=axis,
    )
    output = jnp.concatenate(
        [jnp.unique(sub_array, axis=axis) for sub_array in sub_arrays],
        axis=axis,
    )
    counts = jnp.array([sub_array.shape[axis] for sub_array in sub_arrays])
    inverse_indices = jnp.repeat(jnp.arange(len(counts)), counts)
    if x_shape:
        inverse_indices = jnp.reshape(inverse_indices, x_shape)
    return Results(
        output.astype(x.dtype),
        inverse_indices,
        counts,
    )
