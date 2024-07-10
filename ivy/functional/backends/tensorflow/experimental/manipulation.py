# global
from collections import namedtuple
from typing import (
    Iterable,
    Union,
    Optional,
    Sequence,
    Tuple,
    NamedTuple,
    List,
    Literal,
    Callable,
    Any,
)
from numbers import Number
import tensorflow as tf

# local
from ivy.func_wrapper import with_unsupported_dtypes, handle_out_argument
from .. import backend_version
import ivy
from ivy.functional.ivy.experimental.manipulation import _to_tf_padding


def moveaxis(
    a: Union[tf.Tensor, tf.Variable],
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.moveaxis(a, source, destination)


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16",)}, backend_version)
def heaviside(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.cast(tf.experimental.numpy.heaviside(x1, x2), x1.dtype)


def flipud(
    m: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.flipud(m)


def vstack(
    arrays: Union[Sequence[tf.Tensor], Sequence[tf.Variable]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.vstack(arrays)


def hstack(
    arrays: Union[Sequence[tf.Tensor], Sequence[tf.Variable]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.hstack(arrays)


def rot90(
    m: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[int, int] = (0, 1),
    out: Union[tf.Tensor, tf.Variable] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.rot90(m, k, axes)


@with_unsupported_dtypes({"2.15.0 and below": ("unsigned", "complex")}, backend_version)
def top_k(
    x: tf.Tensor,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    k = min(k, x.shape[axis])
    if not largest:
        indices = tf.experimental.numpy.argsort(x, axis=axis)
        indices = tf.experimental.numpy.take(
            indices, tf.experimental.numpy.arange(k), axis=axis
        )
        indices = tf.dtypes.cast(indices, tf.int32)
    else:
        indices = tf.experimental.numpy.argsort(-x, axis=axis)
        indices = tf.experimental.numpy.take(
            indices, tf.experimental.numpy.arange(k), axis=axis
        )
        indices = tf.dtypes.cast(indices, tf.int32)
    if not sorted:
        indices = tf.sort(indices, axis=axis)
    topk_res = NamedTuple("top_k", [("values", tf.Tensor), ("indices", tf.Tensor)])
    val = tf.experimental.numpy.take_along_axis(x, indices, axis=axis)
    indices = tf.dtypes.cast(indices, tf.int64)
    return topk_res(val, indices)


def fliplr(
    m: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.fliplr(m)


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16",)}, backend_version)
def i0(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.bessel_i0(x, name=None)


def vsplit(
    ary: Union[tf.Tensor, tf.Variable],
    indices_or_sections: Union[int, Sequence[int], tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[tf.Tensor, tf.Variable]]:
    if len(ary.shape) < 2:
        raise ivy.utils.exceptions.IvyError(
            "vsplit only works on arrays of 2 or more dimensions"
        )
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=0)


def dsplit(
    ary: Union[tf.Tensor, tf.Variable],
    indices_or_sections: Union[int, Sequence[int], tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[tf.Tensor, tf.Variable]]:
    if len(ary.shape) < 3:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=2)


def atleast_1d(
    *arys: Union[tf.Tensor, tf.Variable, bool, Number],
    copy: Optional[bool] = None,
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.atleast_1d(*arys)


def dstack(
    arrays: Union[Sequence[tf.Tensor], Sequence[tf.Variable]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.dstack(arrays)


def atleast_2d(
    *arys: Union[tf.Tensor, tf.Variable],
    copy: Optional[bool] = None,
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.atleast_2d(*arys)


def atleast_3d(
    *arys: Union[tf.Tensor, tf.Variable, bool, Number],
    copy: Optional[bool] = None,
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.atleast_3d(*arys)


def take_along_axis(
    arr: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if len(arr.shape) != len(indices.shape):
        raise ivy.utils.exceptions.IvyException(
            "arr and indices must have the same number of dimensions;"
            + f" got {len(arr.shape)} vs {len(indices.shape)}"
        )
    indices = tf.dtypes.cast(indices, tf.int32)
    if mode not in ["clip", "fill", "drop"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes are 'clip', 'fill', 'drop'."
        )
    arr_shape = arr.shape
    if axis < 0:
        axis += len(arr.shape)
    if mode == "clip":
        max_index = arr.shape[axis] - 1
        indices = tf.clip_by_value(indices, 0, max_index)
    elif mode in ("fill", "drop"):
        if "float" in str(arr.dtype) or "complex" in str(arr.dtype):
            fill_value = tf.constant(float("nan"), dtype=arr.dtype)
        elif "uint" in str(arr.dtype):
            fill_value = tf.constant(arr.dtype.max, dtype=arr.dtype)
        elif "int" in str(arr.dtype):
            fill_value = tf.constant(-arr.dtype.max - 1, dtype=arr.dtype)
        else:
            raise TypeError(
                f"Invalid dtype '{arr.dtype}'. Valid dtypes are 'float', 'complex',"
                " 'uint', 'int'."
            )
        indices = tf.where((indices < 0) | (indices >= arr.shape[axis]), -1, indices)
        arr_shape = list(arr_shape)
        arr_shape[axis] = 1
        fill_arr = tf.fill(arr_shape, fill_value)
        arr = tf.concat([arr, fill_arr], axis=axis)
    return tf.experimental.numpy.take_along_axis(arr, indices, axis)


def hsplit(
    ary: Union[tf.Tensor, tf.Variable],
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[tf.Tensor, tf.Variable]]:
    if len(ary.shape) == 1:
        return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=0)
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=1)


def broadcast_shapes(
    *shapes: Union[List[int], List[Tuple]],
) -> Tuple[int, ...]:
    if len(shapes) > 1:
        desired_shape = tf.broadcast_dynamic_shape(shapes[0], shapes[1])
        if len(shapes) > 2:
            for i in range(2, len(shapes)):
                desired_shape = tf.broadcast_dynamic_shape(desired_shape, shapes[i])
    else:
        return [shapes[0]]
    return tuple(desired_shape.numpy().tolist())


def pad(
    input: Union[tf.Tensor, tf.Variable],
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
) -> Union[tf.Tensor, tf.Variable]:
    pad_width = _to_tf_padding(pad_width, len(input.shape))
    if not isinstance(constant_values, (tf.Variable, tf.Tensor)):
        constant_values = tf.constant(constant_values)
    if constant_values.dtype != input.dtype:
        constant_values = tf.cast(constant_values, input.dtype)
    return tf.pad(
        input,
        pad_width,
        mode=mode,
        constant_values=constant_values,
    )


pad.partial_mixed_handler = (
    lambda *args, mode="constant", constant_values=0, reflect_type="even", **kwargs: (
        _check_tf_pad(args[0].shape, args[1], mode, constant_values, reflect_type)
    )
)


def _check_tf_pad(input_shape, pad_width, mode, constant_values, reflect_type):
    pad_width = _to_tf_padding(pad_width, len(input_shape))
    return isinstance(constant_values, Number) and (
        mode == "constant"
        or (
            reflect_type == "even"
            and (
                (
                    mode == "reflect"
                    and all(
                        pad_width[i][0] < s and pad_width[i][1] < s
                        for i, s in enumerate(input_shape)
                    )
                )
                or (
                    mode == "symmetric"
                    and all(
                        pad_width[i][0] <= s and pad_width[i][1] <= s
                        for i, s in enumerate(input_shape)
                    )
                )
            )
        )
    )


def expand(
    x: Union[tf.Tensor, tf.Variable],
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    shape = list(shape)
    n_extra_dims = len(shape) - len(x.shape)
    if n_extra_dims > 0:
        new_shape = (1,) * n_extra_dims + tuple(x.shape)
        x = tf.reshape(x, new_shape)
    for i, dim in enumerate(shape):
        if dim < 0:
            shape[i] = x.shape[i]
    return tf.broadcast_to(x, shape)


def concat_from_sequence(
    input_sequence: Union[Tuple[tf.Tensor], List[tf.Tensor]],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    is_tuple = type(input_sequence) is tuple
    if is_tuple:
        input_sequence = list(input_sequence)
    highest_dtype = input_sequence[0].dtype
    for i in input_sequence:
        highest_dtype = ivy.as_native_dtype(ivy.promote_types(highest_dtype, i.dtype))

    if new_axis == 0:
        ret = tf.concat(input_sequence, axis=axis)
        return ret
    elif new_axis == 1:
        ret = tf.stack(input_sequence, axis=axis)
        return ret


def unique_consecutive(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    Results = namedtuple(
        "Results",
        ["output", "inverse_indices", "counts"],
    )
    x_shape = None
    if axis is None:
        x_shape = x.shape
        x = tf.reshape(x, tf.constant([-1]))
        axis = -1
    ndim = len(x.shape)
    if axis < 0:
        axis += ndim
    splits = (
        tf.where(
            tf.math.reduce_any(
                tf.experimental.numpy.diff(x, axis=axis) != 0,
                axis=tuple(i for i in tf.range(ndim) if i != axis),
            )
        )
        + 1
    )
    if tf.size(splits) > 0:
        sub_arrays = tf.experimental.numpy.split(x, tf.reshape(splits, -1), axis=axis)
    else:
        sub_arrays = [x]
    output = tf.concat(
        [
            tf.raw_ops.UniqueV2(x=sub_array, axis=tf.constant([axis]))[0]
            for sub_array in sub_arrays
        ],
        axis=axis,
    )
    counts = tf.convert_to_tensor([sub_array.shape[axis] for sub_array in sub_arrays])
    inverse_indices = tf.repeat(tf.range(len(counts)), counts)
    if x_shape:
        inverse_indices = tf.reshape(inverse_indices, x_shape)
    return Results(
        tf.cast(output, x.dtype),
        tf.cast(inverse_indices, tf.int64),
        tf.cast(counts, tf.int64),
    )


def take(
    x: Union[int, List, tf.Tensor, tf.Variable],
    indices: Union[int, List, tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    mode: str = "clip",
    fill_value: Optional[Number] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if mode not in ["raise", "wrap", "clip", "fill"]:
        raise ValueError("mode must be one of 'clip', 'raise', 'wrap', or 'fill'")
    if not isinstance(x, (tf.Tensor, tf.Variable)):
        x = tf.constant(x)
    if len(x.shape) == 0:
        x = tf.constant([x])
    if not isinstance(indices, (tf.Tensor, tf.Variable)):
        indices = tf.constant(indices)
    if indices.dtype.is_floating:
        indices = tf.cast(indices, tf.int64)

    # raise
    if mode == "raise":
        mode = "clip"
        if ivy.exists(axis):
            if axis >= len(x.shape):
                raise tf.errors.InvalidArgumentError(
                    None,
                    None,
                    f"Shape must be at least rank {axis+1} but is rank {len(x.shape)}",
                )
            x_shape = x.shape[axis]
        else:
            x_shape = tf.reduce_prod(x.shape)

        bound_check = (indices < -x_shape) | (indices >= x_shape)
        if tf.reduce_any(bound_check):
            if len(indices.shape) == 0:
                raise tf.errors.InvalidArgumentError(
                    None, None, f"index {indices} is not in [-{x_shape}, {x_shape})"
                )
            else:
                first_non_zero = tuple(
                    map(
                        lambda n: n[0].numpy(),
                        tf.experimental.numpy.nonzero(bound_check),
                    )
                )
                raise tf.errors.InvalidArgumentError(
                    None,
                    None,
                    f"indices{list(first_non_zero)} = {indices[first_non_zero]} "
                    f"is not in [-{x_shape}, {x_shape})",
                )

    # clip, wrap
    if mode != "fill":
        ret = tf.experimental.numpy.take(x, indices, axis=axis, mode=mode)
        if ivy.exists(out):
            ivy.inplace_update(out, ret)
        return ret

    # fill
    x_dtype = x.dtype
    if fill_value is None:
        # set according to jax behaviour
        # https://tinyurl.com/66jn68uj
        if x_dtype.is_floating or x_dtype.is_complex:
            # NaN for inexact types
            fill_value = float("NaN")
        else:
            if x_dtype == tf.bool:
                # True for booleans
                fill_value = True
            elif x_dtype.is_unsigned:
                # the largest positive value for unsigned types
                fill_value = x_dtype.max
            else:
                # the largest negative value for signed types
                fill_value = x_dtype.min

    fill_value = tf.constant(fill_value, dtype=x_dtype)
    x_shape = x.shape
    ret = tf.experimental.numpy.take(x, indices, axis=axis, mode="wrap")

    if len(ret.shape) == 0:
        # if scalar, scalar fill (replace)
        if tf.reduce_any(indices != 0):
            ret = fill_value
    else:
        rank = len(x.shape)
        if ivy.exists(axis):
            axis = ((axis % rank) + rank) % rank
            x_shape = x_shape[axis]
        else:
            axis = 0
            x_shape = tf.reduce_prod(x_shape)

        bound_check = tf.constant((indices < -x_shape) | (indices >= x_shape))

        if tf.reduce_any(bound_check):
            if axis > 0:
                bound_check = tf.broadcast_to(
                    bound_check, (*x.shape[:axis], *bound_check.shape)
                )
                end_dim = x.shape[-((rank - axis) - 1) :]
            else:
                end_dim = x.shape[-(rank - 1) :]

            if bound_check.shape != ret.shape:
                slicer = list([Ellipsis] + ([None] * len(end_dim)))
                bound_check = tf.broadcast_to(bound_check[slicer], ret.shape)

            ret = tf.where(bound_check, fill_value[None], ret)

    if ivy.exists(out):
        ivy.inplace_update(out, ret)
    return ret


def trim_zeros(a: tf.Tensor, /, *, trim: Optional[str] = "bf") -> tf.Tensor:
    nonzero_indices = tf.where(a != 0)
    first = tf.reduce_min(nonzero_indices)
    last = tf.reduce_max(nonzero_indices) + 1

    trim = trim.upper()
    if "F" in trim:
        first = tf.maximum(first, 0)
    if "B" in trim:
        last = tf.minimum(last, tf.cast(tf.shape(a)[0], tf.int64))

    return a[first:last]


@handle_out_argument
def unflatten(
    x: tf.Tensor,
    /,
    shape: Tuple[int] = None,
    dim: Optional[int] = 0,
    *,
    out: Optional[tf.Tensor] = None,
    name: Optional[str] = None,
) -> tf.Tensor:
    dim = abs(len(x.shape) + dim) if dim < 0 else dim

    # infer the size of any dimensions that are -1
    tf_shape = tf.constant(shape)
    inferred_size = tf.reduce_prod(tf.shape(x)[dim]) // tf.reduce_prod(
        tf.where(tf_shape != -1, x=shape, y=tf.constant(1))
    )
    shape = tf.where(tf_shape != -1, x=shape, y=inferred_size)

    res_shape = x.shape[:dim] + tf.TensorShape(shape) + x.shape[dim + 1 :]
    res = tf.reshape(x, res_shape, name)
    return res
