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
    return ret


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")},
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
    {"2.5.0 and below": ("float16", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def nanmedian(x, axis=None, keepdim=True, name=None):
    return ivy.nanmedian(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.1 and below": ("bool", "float16", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def numel(x, name=None):
    prod = ivy.prod(x.size, dtype=ivy.int64)
    try:
        length = len(x)
    except (ValueError, TypeError):
        length = 1  # if 0 dimensional tensor with 1 element
    return ivy.array(prod if prod > 0 else ivy.array(length, dtype=ivy.int64))


@with_supported_dtypes({"float32", "float64"})
@to_ivy_arrays_and_back
def quantile(x, q, axis=None, keepdim=False, name=None):
    x = (
        ivy.cast(x, ivy.float64)
        if ivy.dtype(x) == "float64"
        else ivy.cast(x, ivy.float32)
    )
    sorted_x = ivy.sort(x, axis=axis)
    rank = (ivy.shape(sorted_x, axis=axis) - 1) * q
    rank_int = ivy.floor(rank)
    rank_frac = rank - rank_int
    lower_values = ivy.gather(sorted_x, ivy.cast(rank_int, ivy.int32), axis=axis)
    upper_values = ivy.gather(sorted_x, ivy.cast(rank_int + 1, ivy.int32), axis=axis)
    quantile_value = lower_values + rank_frac * (upper_values - lower_values)
    if not keepdim:
        quantile_value = ivy.squeeze(quantile_value, axis=axis)

    return quantile_value


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "uint16")},
    "paddle",
)
@to_ivy_arrays_and_back
def std(x, axis=None, unbiased=True, keepdim=False, name=None):
    x = (
        ivy.astype(x, ivy.float64)
        if ivy.dtype(x) == "float64"
        else ivy.astype(x, ivy.float32)
    )
    return ivy.std(x, axis=axis, correction=int(unbiased), keepdims=keepdim)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def var(x, axis=None, unbiased=True, keepdim=False, name=None):
    if unbiased:
        correction = 1
    else:
        correction = 0
    return ivy.var(x, axis=axis, correction=correction, keepdims=keepdim)
