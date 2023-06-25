import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    return ivy.indices(dimensions, dtype=dtype, sparse=sparse)


# unravel_index
@to_ivy_arrays_and_back
def unravel_index(indices, shape, order="C"):
    ret = [x.astype("int64") for x in ivy.unravel_index(indices, shape)]
    return tuple(ret)


@to_ivy_arrays_and_back
def diag_indices(n, ndim=2):
    idx = ivy.arange(n)
    res = ivy.array((idx,) * ndim)
    res = tuple(res.astype("int64"))
    return res


@to_ivy_arrays_and_back
def mask_indices(n, mask_func, k=0):
    a = ivy.ones((n, n))
    mask = mask_func(a, k=k)
    mask = ivy.array(mask)
    indices = ivy.argwhere(mask)
    return indices[:, 0], indices[:, 1]


@to_ivy_arrays_and_back
def tril_indices(n, k=0, m=None):
    return ivy.tril_indices(n, m, k)
