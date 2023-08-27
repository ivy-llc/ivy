import ivy
from ivy.functional.frontends.builtins.func_wrapper import (
    from_zero_dim_arrays_to_scalar,
)


@from_zero_dim_arrays_to_scalar
def radians(x):
    return ivy.deg2rad(x)


@from_zero_dim_arrays_to_scalar
def degrees(x):
    return ivy.rad2deg(x)
