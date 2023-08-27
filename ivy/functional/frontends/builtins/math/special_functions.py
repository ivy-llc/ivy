import ivy
from ivy.functional.frontends.builtins.func_wrapper import (
    from_zero_dim_arrays_to_scalar,
)


@from_zero_dim_arrays_to_scalar
def erf(x):
    return ivy.erf(x)


@from_zero_dim_arrays_to_scalar
def erfc(x):
    return ivy.subtract(1.0, ivy.erf(x))


@from_zero_dim_arrays_to_scalar
def lgamma(x):
    return ivy.lgamma(x)
