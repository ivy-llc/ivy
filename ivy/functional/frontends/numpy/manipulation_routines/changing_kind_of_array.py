# local
import ivy


def asarray(obj,
            dtype=None,
            order=None,
            *,
            like=None):
    if dtype:
        obj = ivy.astype(ivy.array(obj), ivy.as_ivy_dtype(dtype))
    return ivy.asarray(obj, dtype=dtype)