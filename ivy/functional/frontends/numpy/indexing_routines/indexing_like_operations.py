import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    inputs_to_ivy_arrays,
)


@to_ivy_arrays_and_back
def take_along_axis(arr, indices, axis):
    return ivy.take_along_axis(arr, indices, axis)


@to_ivy_arrays_and_back
def tril_indices(n, k=0, m=None):
    return ivy.tril_indices(n, m, k)


@to_ivy_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = tuple()
    else:
        res = ivy.empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = ivy.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1 :])
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


# unravel_index
@to_ivy_arrays_and_back
def unravel_index(indices, shape, order="C"):
    ret = [x.astype("int64") for x in ivy.unravel_index(indices, shape)]
    return tuple(ret)


@to_ivy_arrays_and_back
def fill_diagonal(a, val, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not ivy.all(ivy.diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + ivy.sum(ivy.cumprod(a.shape[:-1]))

    # Write the value out into the diagonal.
    shape = a.shape
    a = ivy.reshape(a, a.size)
    a[:end:step] = val
    a = ivy.reshape(a, shape)


@inputs_to_ivy_arrays
def put_along_axis(arr, indices, values, axis):
    ivy.put_along_axis(arr, indices, values, axis)


def diag(v, k=0):
    return ivy.diag(v, k=k)


@to_ivy_arrays_and_back
def diagonal(a, offset, axis1, axis2):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)



@inputs_to_ivy_arrays
def place(arr, mask, vals):
    arr_shape = arr.shape

    if arr.shape != mask.shape:
        raise ValueError("mask must have the same size as arr")

    vals_new = ivy.reshape(vals, shape=(-1,), order="C")

    # Reshape vals to have the same shape as arr and mask
    total_size = 1
    for diff_size in arr_shape:
        total_size *= diff_size
        if diff_size < 0:
            raise ValueError("values must not be negative")
    if vals_new.size == 0 or total_size == 0:
        return ivy.zeros_like(vals_new)
    repetition = -(-total_size // len(vals_new))
    conc = (vals_new,) * repetition
    vals_new = ivy.concat(conc)[:total_size]
    vals_reshape = ivy.reshape(vals_new, shape=arr_shape, order="C")

    arr = ivy.where(mask, vals_reshape, arr)
