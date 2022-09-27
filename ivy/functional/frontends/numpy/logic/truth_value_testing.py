# global
import ivy


def all(
    a,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=True,
):
    ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


def any(
    a,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=True,
):
    ret = ivy.any(a, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


def isscalar(element):
    return isinstance(element, int) or isinstance(
        element, bool) or isinstance(element, float)
