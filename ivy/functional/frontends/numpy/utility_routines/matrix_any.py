import ivy


def any(
    x,
    /,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=True
):
    ret = ivy.where(ivy.array(where), ivy.array(x), ivy.zeros_like(x))
    ret = ivy.any(ret, axis=axis, keepdims=keepdims, out=out)
    if len(ret.shape) == 0:
        ret = bool(ret) 
    return ret
 
