import inspect

import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def diag_indices(n, ndim=2):
    idx = ivy.arange(n)
    res = ivy.array((idx,) * ndim)
    res = tuple(res.astype("int64"))
    return res


@to_ivy_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    return ivy.indices(dimensions, dtype=dtype, sparse=sparse)


@to_ivy_arrays_and_back
def mask_indices(n, mask_func, k=0):
    mask_func_obj = inspect.unwrap(mask_func)
    mask_func_name = mask_func_obj.__name__
    try:
        ivy_mask_func_obj = getattr(ivy.functional.frontends.numpy, mask_func_name)
        a = ivy.ones((n, n))
        mask = ivy_mask_func_obj(a, k=k)
        indices = ivy.argwhere(mask.ivy_array)
        ret = indices[:, 0], indices[:, 1]
        return tuple(ret)
    except AttributeError as e:
        print(f"Attribute error: {e}")


@to_ivy_arrays_and_back
def tril_indices(n, k=0, m=None):
    return ivy.tril_indices(n, m, k)


@to_ivy_arrays_and_back
def tril_indices_from(arr, k=0):
    return ivy.tril_indices(arr.shape[0], arr.shape[1], k)


# unravel_index
@to_ivy_arrays_and_back
def unravel_index(indices, shape, order="C"):
    ret = [x.astype("int64") for x in ivy.unravel_index(indices, shape)]
    return tuple(ret)
