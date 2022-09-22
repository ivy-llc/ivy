# global
import ivy


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
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    if ivy.is_array(where):
        a = ivy.where(where, a, ivy.default(out, ivy.zeros_like(a)), out=out)
    if initial is not None:
        s = ivy.shape(a, as_array=True)
        s[axis] = 1
        header = ivy.full(ivy.Shape(tuple(s)), initial)
        a = ivy.concat([a, header], axis=axis)

    return ivy.min(a, axis=axis, keepdims=keepdims, out=out)
