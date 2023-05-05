import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def put(a, ind, v, mode="raise"):
    shape = a.shape
    ind_flat = ivy.flatten(ind)
    flat = ivy.flatten(a)
    length = len(flat)

    if mode == "raise":
        if not ivy.all(ind_flat < length):
            raise ValueError("indices out of bounds")

        if not ivy.all(v < length):
            raise ValueError("values out of bounds")

    elif mode == "wrap":
        ind_flat = ind_flat % length

    elif mode == "clip":
        ind_flat = ivy.clip(ind_flat, 0, length - 1)

    # repeats v if size mismatch
    length_v = len(v)
    length_ind_flat = len(ind_flat)
    repeats = -(-length_v // length_ind_flat)  # ceil division
    v = ivy.concat(ivy.flatten(v) * repeats)[:length_v]

    flat[ind_flat] = v

    return ivy.reshape(flat, shape)


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
    temp = ivy.flatten(a)
    temp[:end:step] = val
    a = ivy.reshape(temp, shape)
