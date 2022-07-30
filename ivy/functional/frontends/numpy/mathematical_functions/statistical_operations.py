import ivy

def std(
    x,
    axis=None,
    correction=0.0,
    keepdims=False,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.std(x1, axis, correction, keepdims, out=out)
    if x.size == 0:
        ret = 0
    if ivi.isnan(x):
        ret = ivi.nan
    if dtype:
        x1 = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

