# global
import ivy


def around(a, decimals=0, out=None):
    ten_raised = 10.0 ** decimals
    return ivy.divide(rint(ivy.multiply(a, ten_raised)), ten_raised, out=out)


def round_(a, decimals=0, out=None):
    return around(a, decimals=decimals, out=out)


def rint(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), 
        ivy.round(x), 
        ivy.default(out, x), 
        out=out
    )
    return ret


def fix(
    x,
    /,
    out=None,
):
    where = ivy.greater_equal(x, 0)
    return ivy.where(where, ivy.floor(x, out=out), ivy.ceil(x, out=out), out=out)


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
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), 
        ivy.floor(x), 
        ivy.default(out, x), 
        out=out
    )
    return ret


def ceil(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), 
        ivy.ceil(x), 
        ivy.default(out, x), 
        out=out
    )
    return ret


def trunc(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), 
        ivy.trunc(x), 
        ivy.default(out, x), 
        out=out
    )
    return ret
