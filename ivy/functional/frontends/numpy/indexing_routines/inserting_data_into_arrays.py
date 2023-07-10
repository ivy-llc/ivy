import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


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


@to_ivy_arrays_and_back
def put(a, ind, v, mode="raise"):
    if not ivy.is_array(ind):
        ind = ivy.array([ind])
    if not ivy.is_array(a) == int:
        v = ivy.array([v])

    a_shape = ivy.shape(a)
    a_length = int(ivy.prod(a_shape))
    a_flat = ivy.flatten(a)
    ind_length = int(ivy.prod(ivy.shape(ind)))
    v_length = int(ivy.prod(ivy.shape(v)))

    if mode == "raise":
        for i in range(ind_length):
            if ind[i] >= a_length:
                raise IndexError(
                    "index "
                    + str(ind[i])
                    + " is out of bounds for axis 0 with size "
                    + str(a_length)
                )
            else:
                a_flat[ind[i]] = v[i % v_length]

    if mode == "wrap":
        for i in range(ind_length):
            a_flat[ind[i] % a_length] = v[i % v_length]

    if mode == "clip":
        for i in range(ind_length):
            if ind[i] < a_length:
                a_flat[int(max(0, ind[i]))] = v[i % v_length]
            else:
                a_flat[int(min((a_length - 1), ind[i]))] = v[i % v_length]

    a.reshape(a_shape)
