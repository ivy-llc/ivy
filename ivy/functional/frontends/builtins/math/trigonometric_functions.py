import ivy
from ivy.functional.frontends.builtins.func_wrapper import (
    from_zero_dim_arrays_to_scalar,
    to_ivy_arrays_and_back,
)


@from_zero_dim_arrays_to_scalar
def acos(x):
    return ivy.acos(x)


@from_zero_dim_arrays_to_scalar
def asin(x):
    return ivy.asin(x)


@from_zero_dim_arrays_to_scalar
def atan(x):
    return ivy.atan(x)


@from_zero_dim_arrays_to_scalar
def atan2(x, y):
    return ivy.atan2(x, y)


@from_zero_dim_arrays_to_scalar
def cos(x):
    return ivy.cos(x)


@from_zero_dim_arrays_to_scalar
def sin(x):
    return ivy.sin(x)


@from_zero_dim_arrays_to_scalar
def tan(x):
    return ivy.tan(x)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def dist(p, q):
    return ivy.vector_norm(ivy.subtract(p, q))


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def hypot(*coordinates):
    return ivy.vector_norm(coordinates)
