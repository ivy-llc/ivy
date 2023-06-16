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
    if weights is None:
        weights = ivy.ones_like(a)
    if range is None:
        range = (a.min(), a.max())
    if density is None:
        density = False
    return ivy.histogram(
        a,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
    )
