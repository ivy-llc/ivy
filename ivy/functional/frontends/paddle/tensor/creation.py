# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from .tensor import Tensor
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def to_tensor(data, /, *, dtype=None, place=None, stop_gradient=True):
    array = ivy.array(data, dtype=dtype, device=place)
    return Tensor(array, dtype=dtype, place=place)


@with_unsupported_dtypes({"2.4.2 and below": "int8"}, "paddle")
@to_ivy_arrays_and_back
def ones(shape, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return ivy.ones(shape, dtype=dtype)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("uint8", "int8", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def ones_like(x, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return ivy.ones_like(x, dtype=dtype)


@with_unsupported_dtypes({"2.4.2 and below": "int8"}, "paddle")
@to_ivy_arrays_and_back
def zeros(shape, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return ivy.zeros(shape, dtype=dtype)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("uint8", "int8", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def zeros_like(x, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return ivy.zeros_like(x, dtype=dtype)


@to_ivy_arrays_and_back
def full(shape, fill_value, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return ivy.full(shape, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def full_like(x, fill_value, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return ivy.full_like(x, fill_value, dtype=dtype)
