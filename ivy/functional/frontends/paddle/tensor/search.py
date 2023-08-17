# global
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_ivy_arrays_and_back
def argmax(x, /, *, axis=None, keepdim=False, dtype="int64", name=None):
    return ivy.argmax(x, axis=axis, keepdims=keepdim, dtype=dtype)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_ivy_arrays_and_back
def argmin(x, /, *, axis=None, keepdim=False, dtype="int64", name=None):
    return ivy.argmin(x, axis=axis, keepdims=keepdim, dtype=dtype)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_ivy_arrays_and_back
def argsort(x, /, *, axis=-1, descending=False, name=None):
    return ivy.argsort(x, axis=axis, descending=descending)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def sort(x, /, *, axis=-1, descending=False, name=None):
    return ivy.sort(x, axis=axis, descending=descending)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_ivy_arrays_and_back
def nonzero(input, /, *, as_tuple=False):
    ret = ivy.nonzero(input)
    if as_tuple is False:
        ret = ivy.matrix_transpose(ivy.stack(ret))
    return ret


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def searchsorted(sorted_sequence, values, out_int32=False, right=False, name=None):
    if right:
        side = "right"
    else:
        side = "left"
    ret = ivy.searchsorted(sorted_sequence, values, side=side)
    if out_int32:
        ret = ivy.astype(ret, "int32")
    return ret


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def masked_select(x, mask, name=None):
    return ivy.flatten(x[mask])


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
def topk(x, k, axis=None, largest=True, sorted=True, name=None):
    return ivy.top_k(x, k, axis=axis, largest=largest, sorted=sorted)


@with_supported_dtypes(
    {
        "2.5.1 and below": (
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        )
    },
    "paddle",
)
def where(condition, x=None, y=None, name=None):
    if ivy.isscalar(x):
        x = ivy.full([1], x, ivy.asarray([x]).dtype.name)

    if ivy.isscalar(y):
        y = ivy.full([1], y, ivy.asarray([y]).dtype.name)

    if x is None and y is None:
        return nonzero(condition, as_tuple=True)

    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")

    condition_shape = list(condition.shape)
    x_shape = list(x.shape)
    y_shape = list(y.shape)

    if x_shape == y_shape and condition_shape == x_shape:
        broadcast_condition = condition
        broadcast_x = x
        broadcast_y = y
    else:
        zeros_like_x = ivy.zeros_like(x)
        zeros_like_y = ivy.zeros_like(y)
        zeros_like_condition = ivy.zeros_like(condition)
        zeros_like_condition = ivy.cast(zeros_like_condition, x.dtype)
        cast_cond = ivy.cast(condition, x.dtype)

        broadcast_zeros = ivy.add(zeros_like_x, zeros_like_y)
        broadcast_zeros = ivy.add(broadcast_zeros, zeros_like_condition)
        broadcast_x = ivy.add(x, broadcast_zeros)
        broadcast_y = ivy.add(y, broadcast_zeros)
        broadcast_condition = ivy.add(cast_cond, broadcast_zeros)
        broadcast_condition = ivy.cast(broadcast_condition, "bool")

    ret = ivy.where(broadcast_condition, broadcast_x, broadcast_y)
    return ret
