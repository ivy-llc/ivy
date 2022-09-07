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
    pass


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
