# global
import ivy


def cos(
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
        ivy.broadcast_to(where, x.shape), ivy.cos(x), ivy.default(out, x), out=out
    )
    return ret


cos.unsupported_dtypes = {"torch": ("float16",)}


def sin(
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
        ivy.broadcast_to(where, x.shape), ivy.sin(x), ivy.default(out, x), out=out
    )
    return ret


sin.unsupported_dtypes = {"torch": ("float16",)}


def tan(
    x,
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.tan(x), ivy.default(out, x), out=out
    )
    return ret


tan.unsupported_dtypes = {"torch": ("float16",)}


def arcsin(
    x,
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.asin(x), ivy.default(out, x), out=out
    )
    return ret


arcsin.unsupported_dtypes = {"torch": ("float16",)}


def arccos(
    x,
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.acos(x), ivy.default(out, x), out=out
    )
    return ret


arcsin.unsupported_dtypes = {"torch": ("float16",)}


def arctan(
    x,
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.atan(x), ivy.default(out, x), out=out
    )
    return ret


arctan.unsupported_dtypes = {"torch": ("float16",)}


def hypot(
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
    ret = ivy.where(
        ivy.broadcast_to(where, x1.shape), 
        ivy.sqrt(ivy.add(ivy.square(x1), ivy.square(x2))), 
        ivy.default(out, x1), 
        out=out
    )
    return ret


def arctan2(
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
    ret = ivy.where(
        ivy.broadcast_to(where, x1.shape), 
        ivy.atan2(x1, x2), 
        ivy.default(out, x1), 
        out=out
    )
    return ret


def degrees(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    return rad2deg(
        x, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok
    )


def radians(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    return deg2rad(
        x, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok
    )


def unwrap(
    p, 
    discont=None, 
    axis=-1,
    *,
    period=6.283185307179586
):
    p = ivy.array(p)
    ndim = p.ndim
    all_but_last = ivy.arange(0, p.shape[axis] - 1)
    all_but_first = ivy.arange(1, p.shape[axis])
    dd = ivy.subtract(
        ivy.gather(p, all_but_first, axis=axis),
        ivy.gather(p, all_but_last, axis=axis)
    )
    if discont is None:
        discont = period / 2
    slice1 = [slice(None, None)] * ndim
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    if ivy.is_int_dtype(p.dtype):
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = ivy.remainder(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        # for `mask = (abs(dd) == period/2)`, the above line made
        # `ddmod[mask] == -period/2`. correct these such that
        # `ddmod[mask] == sign(dd[mask])*period/2`.
        ivy.where(
            ivy.logical_and(ddmod == interval_low, dd > 0), 
            interval_high, 
            ddmod,
            out=ddmod
        )
    ph_correct = ddmod - dd
    ivy.where(ivy.less(ivy.abs(dd), discont), ph_correct, 0)
    up = ivy.array(p, copy=True, dtype=p.dtype)
    up[slice1] = p[slice1] + ivy.cumsum(ph_correct, axis)
    return up


def deg2rad(
    x,
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), 
        ivy.multiply(x, ivy.pi / 180.0), 
        ivy.default(out, x), 
        out=out
    )
    return ret


def rad2deg(
    x,
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(
        ivy.broadcast_to(ivy.array(where), x.shape), 
        ivy.multiply(x, 180.0 / ivy.pi), 
        ivy.default(out, x), 
        out=out
    )
    return ret
