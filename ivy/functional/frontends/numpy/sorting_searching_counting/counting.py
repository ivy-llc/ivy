# global
import ivy


def count_nonzero(a, axis=None, *, keepdims=False):
    x = ivy.array(a)
    zero = ivy.zeros(ivy.shape(x), dtype=x.dtype)
    return ivy.sum(
        ivy.astype(ivy.not_equal(x, zero), ivy.int64),
        axis=axis,
        keepdims=keepdims,
    )
