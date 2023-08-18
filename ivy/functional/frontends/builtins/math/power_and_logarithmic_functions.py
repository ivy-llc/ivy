import ivy
from ivy.functional.frontends.builtins.func_wrapper import (
    from_zero_dim_arrays_to_scalar,
)


@from_zero_dim_arrays_to_scalar
def sqrt(x):
    return ivy.sqrt(x)


@from_zero_dim_arrays_to_scalar
def pow(x, y):
    return ivy.pow(x, y)


@from_zero_dim_arrays_to_scalar
def log(x, base=None):
    if base:
        return ivy.divide(ivy.log(x), ivy.log(base))
    return ivy.log(x)


@from_zero_dim_arrays_to_scalar
def log2(x):
    return ivy.log2(x)


@from_zero_dim_arrays_to_scalar
def exp(x):
    return ivy.exp(x)


@from_zero_dim_arrays_to_scalar
def exp2(x):
    return ivy.exp2(x)


@from_zero_dim_arrays_to_scalar
def expm1(x):
    return ivy.expm1(x)


@from_zero_dim_arrays_to_scalar
def log1p(x):
    return ivy.log1p(x)


@from_zero_dim_arrays_to_scalar
def log10(x):
    return ivy.log10(x)
