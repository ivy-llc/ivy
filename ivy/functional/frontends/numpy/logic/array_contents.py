# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    inputs_to_ivy_arrays,
    from_zero_dim_arrays_to_scalar,
)
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def isneginf(x, out=None):
    isinf = ivy.isinf(x)
    neg_sign_bit = ivy.less(x, 0)
    return ivy.logical_and(isinf, neg_sign_bit, out=out)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def isposinf(x, out=None):
    isinf = ivy.isinf(x)
    pos_sign_bit = ivy.greater(x, 0)
    return ivy.logical_and(isinf, pos_sign_bit, out=out)


@inputs_to_ivy_arrays
@from_zero_dim_arrays_to_scalar
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@inputs_to_ivy_arrays
@from_zero_dim_arrays_to_scalar
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
