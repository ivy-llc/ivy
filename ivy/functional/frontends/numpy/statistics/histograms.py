import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def histogram(
    a,
    bins=10,
    range=None,
    normed=None,
    weights=None,
    density=None,
    *,
    out=None,
    dtype=None,
):
    if dtype:
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
    return ivy.histogram(
        a,
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
        out=out,
    )
