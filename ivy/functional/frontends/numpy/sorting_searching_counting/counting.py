# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def count_nonzero(a, axis=None, *, keepdims=False):
    x = ivy.array(a)
    zero = ivy.zeros(ivy.shape(x), dtype=x.dtype)
    return ivy.sum(
        ivy.astype(ivy.not_equal(x, zero), ivy.int64),
        axis=axis,
        keepdims=keepdims,
    )
