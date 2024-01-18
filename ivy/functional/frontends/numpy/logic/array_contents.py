# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    inputs_to_ivy_arrays,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs
from ivy.func_wrapper import with_supported_dtypes


@inputs_to_ivy_arrays
@from_zero_dim_arrays_to_scalar
def allclose(a, b, /, *, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def isclose(a, b, /, *, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@with_supported_dtypes(
    {"2.6.0 and below": ("int64", "float64", "float32", "int32", "bfloat16")}, "paddle"
)
@to_ivy_arrays_and_back
def isin(element, test_elements, assume_unique=False, invert=False):
    return ivy.isin(element, test_elements, assume_unique=assume_unique, invert=invert)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def isneginf(x, out=None):
    return ivy.isinf(x, detect_positive=False)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def isposinf(x, out=None):
    return ivy.isinf(x, detect_negative=False)
