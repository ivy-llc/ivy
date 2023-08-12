# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)

@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def mean(input, axis=None, keepdim=False, out=None):
    ret = ivy.mean(input, axis=axis, keepdims=keepdim, out=out)
    ret = ivy.expand_dims(ret, axis=-1) if ret.ndim == 0 else ret
    return ret

@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def var(x, axis=None, keepdim=False, name=None):
    return ivy.variance(x, axis=axis, keepdims=keepdim)

@with_unsupported_dtypes({"2.5.1 and below": ("complex", "int8")}, "paddle")
@to_ivy_arrays_and_back
def numel(x, name=None):
    prod = ivy.prod(x.size, dtype=ivy.int64)
    try:
        length = len(x)
    except (ValueError, TypeError):
        length = 1  # if 0 dimensional tensor with 1 element
    return ivy.array(prod if prod > 0 else ivy.array(length, dtype=ivy.int64))


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def nanquantile(a, q, axis=None, keepdims=False, interpolation="linear", out=None):
    return ivy.nanquantile(
        a, q, axis=axis, keepdims=keepdims, interpolation=interpolation, out=out
    )


@with_supported_dtypes(
    {"2.5.1 and below": ("bool", "float16", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def median(x, axis=None, keepdim=False, name=None):
    x = (
        ivy.astype(x, ivy.float64)
        if ivy.dtype(x) == "float64"
        else ivy.astype(x, ivy.float32)
    )
    return ivy.median(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.0 and below": ("float16", "float32", "float64", "uint16")},
    "paddle",
)
@to_ivy_arrays_and_back
def nanmedian(x, axis=None, keepdim=True, name=None):
    x = (
        ivy.astype(x, ivy.float64)
        if ivy.dtype(x) == "float64"
        else ivy.astype(x, ivy.float32)
    )
    return ivy.median(x, axis=axis, keepdims=keepdim)


