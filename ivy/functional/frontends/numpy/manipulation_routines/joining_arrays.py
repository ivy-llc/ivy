# local
import ivy


def concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting="same_kind"):
    if dtype:
        arrays = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in arrays]
    return ivy.concat(arrays, axis, out=out)
