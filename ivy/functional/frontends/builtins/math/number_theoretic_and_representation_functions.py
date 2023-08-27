import ivy
from ivy.functional.frontends.builtins.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@from_zero_dim_arrays_to_scalar
def ceil(x):
    return ivy.ceil(x).astype(int)


@from_zero_dim_arrays_to_scalar
def copysign(x, y):
    return ivy.copysign(x, y)


@from_zero_dim_arrays_to_scalar
def fabs(x):
    return ivy.abs(x).astype(float)


@from_zero_dim_arrays_to_scalar
def floor(x):
    return ivy.floor(x).astype(int)


@to_ivy_arrays_and_back
def fmod(x, y):
    return ivy.fmod(x, y).astype(float)


@from_zero_dim_arrays_to_scalar
def frexp(x):
    return ivy.frexp(x)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def fsum(x):
    return ivy.sum(x).astype(float)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def gcd(*integers):
    if len(integers) == 0:
        return 0
    return ivy.gcd(*integers)


@from_zero_dim_arrays_to_scalar
def isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0):
    return ivy.isclose(a, b, rtol=rel_tol, atol=abs_tol)


@from_zero_dim_arrays_to_scalar
def isfinite(x):
    return ivy.isfinite(x)


@from_zero_dim_arrays_to_scalar
def isinf(x):
    return ivy.isinf(x)


@from_zero_dim_arrays_to_scalar
def isnan(x):
    return ivy.isnan(x)


@from_zero_dim_arrays_to_scalar
def isqrt(x):
    return ivy.floor(ivy.sqrt(x)).astype(int)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def lcm(*integers):
    if len(integers) == 0:
        return 1
    return ivy.lcm(*integers)


@from_zero_dim_arrays_to_scalar
def ldexp(x, i):
    return ivy.ldexp(x, i)


@from_zero_dim_arrays_to_scalar
def modf(x):
    return ivy.modf(x)


@from_zero_dim_arrays_to_scalar
def nextafter(x, y):
    return ivy.nextafter(x, y)


@from_zero_dim_arrays_to_scalar
def remainder(x, y):
    return ivy.remainder(x, y).astype(float)


@from_zero_dim_arrays_to_scalar
def trunc(x):
    return ivy.trunc(x)
