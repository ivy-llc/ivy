import ivy
from ivy.functional.frontends.builtins.func_wrapper import (
    from_zero_dim_arrays_to_scalar,
)


@from_zero_dim_arrays_to_scalar
def acosh(x):
    return ivy.acosh(x)


@from_zero_dim_arrays_to_scalar
def asinh(x):
    return ivy.asinh(x)


@from_zero_dim_arrays_to_scalar
def atanh(x):
    return ivy.atanh(x)


@from_zero_dim_arrays_to_scalar
def cosh(x):
    return ivy.cosh(x)


@from_zero_dim_arrays_to_scalar
def sinh(x):
    return ivy.sinh(x)


@from_zero_dim_arrays_to_scalar
def tanh(x):
    return ivy.tanh(x)
