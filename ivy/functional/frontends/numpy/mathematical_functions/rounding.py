import ivy


def floor(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
    signature=None,
    extobj=None,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.floor(x, out=out)
    if ivy.is_array(where):
        if out is None:
            out = ivy.empty(ret.shape)
        ret = ivy.where(where, ret, out, out=out)
    return ret
