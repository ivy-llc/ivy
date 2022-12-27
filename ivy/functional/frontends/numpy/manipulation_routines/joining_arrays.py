# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting="same_kind"):
    if dtype:
        arrays = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in arrays]
    return ivy.concat(arrays, axis=axis, out=out)


@to_ivy_arrays_and_back
def stack(arrays, axis=0, out=None):
    return ivy.stack(arrays, axis=axis, out=out)


@to_ivy_arrays_and_back
def vstack(tup):
    if len(ivy.shape(tup[0])) == 1:
        xs = []
        for t in tup:
            xs += [ivy.reshape(t, (1, ivy.shape(t)[0]))]
        return ivy.concat(xs, axis=0)
    return ivy.concat(tup, axis=0)


row_stack = vstack


@to_ivy_arrays_and_back
def hstack(tup):
    if len(ivy.shape(tup[0])) == 1:
        xs = []
        for t in tup:
            xs += [ivy.reshape(t, (1, ivy.shape(t)[0]))]
        ret = ivy.concat(xs, axis=-1)
    else:
        ret = ivy.concat(tup, axis=1)
    return ret
