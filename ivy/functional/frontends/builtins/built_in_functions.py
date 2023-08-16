import ivy
from ivy.functional.frontends.builtins.func_wrapper import (
    from_zero_dim_arrays_to_scalar,
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def abs(x):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def range(start, /, stop=None, step=1):
    if not stop:
        return ivy.arange(0, stop=start, step=step)
    return ivy.arange(start, stop=stop, step=step)
