import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def histogram(
    a,
    *,
    bins=10,
    range=None,
    density=None,
    weights=None,
):
    return ivy.histogram(
        a,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
    )
