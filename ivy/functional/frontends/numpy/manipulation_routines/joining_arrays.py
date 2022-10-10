# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_casting


@handle_numpy_casting
def concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting="same_kind"):
    if dtype:
        arrays = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in arrays]
    return ivy.concat(arrays, axis=axis, out=out)
