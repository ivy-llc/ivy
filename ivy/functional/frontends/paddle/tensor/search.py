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
    {"2.5.0 and below": ("float32", "float64", "int32", "int64")},
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
def topk(x, k, axis=None, largest=True, sorted=True, name=None):
    return ivy.top_k(x, k, axis=axis, largest=largest, sorted=sorted)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def mode(x, axis=None, keepdim=False, name=None):
    return ivy.indices(x, axis, keepdim, sorted=sorted)
