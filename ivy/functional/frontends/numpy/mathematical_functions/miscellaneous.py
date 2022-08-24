import ivy

def clip(a, a_min, a_max, /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,):
    if dtype:
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
    if a_min > a_max:
        ret = a.full_like(a_max)
    else:
        ret = ivy.clip(a, a_min, a_max, out=out)
        
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret

