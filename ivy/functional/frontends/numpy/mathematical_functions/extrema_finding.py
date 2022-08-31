# global
import ivy
import numpy as np


def minimum(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    if dtype:
        x1 = ivy.astype(ivy.array(x1), ivy.as_ivy_dtype(dtype))
        x2 = ivy.astype(ivy.array(x2), ivy.as_ivy_dtype(dtype))
    ret = ivy.minimum(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


def amin(
    a,
    axis=None,
    out=None,
    keepdims=None,
    initial=None,
    where=None, 
):
    a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(ivy.dtype(a[0])))
    ret=np.amin(a, axis=axis, out=out, keepdims=keepdims, initial=initial,where=where)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret
