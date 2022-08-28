# global
import math
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


def convolve(a, v, mode='full'):
    pass


@from_zero_dim_arrays_to_float
def clip(a, 
         a_min, 
         a_max, 
         /,
         out=None,
         *,
         where=True,
         casting="same_kind",
         order="k",
         dtype=None,
         subok=True,):
    
    if not dtype:
        dtype = a.dtype
    ret = ivy.where(ivy.broadcast_to(where, a.shape), ivy.clip(a, a_min, a_max), ivy.default(out, a), out=out)
    return ret


# sqrt
@from_zero_dim_arrays_to_float
def sqrt(
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
    x = ivy.array(x)
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.where(ivy.broadcast_to(where, x.shape), ivy.sqrt(x), ivy.default(out, x), out=out)
    return ret


@from_zero_dim_arrays_to_float
def cbrt(
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
        x =  ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    all_positive = ivy.pow(ivy.abs(x), 1.0 / 3.0)
    correct_signs = ivy.where(ivy.less(x, 0.0), ivy.negative(all_positive), all_positive) 
    ret = ivy.where(ivy.broadcast_to(where, x.shape), correct_signs, ivy.default(out, x), out=out)
    return ret


def square(
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
    ret = ivy.where(ivy.broadcast_to(where, x.shape), ivy.square(x), ivy.default(out, x), out=out)
    return ret


def absolute(
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
    ret = ivy.where(ivy.broadcast_to(where, x.shape), ivy.abs(x), ivy.default(out, x), out=out)
    return ret


def fabs(
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
    ret = ivy.where(ivy.broadcast_to(where, x.shape), ivy.abs(x), ivy.default(out, x), out=out)
    return ret


def sign(
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
    ret = ivy.sign(x, out=out)
    if where is not None:
        ret = ivy.where(ivy.broadcast_to(where, x.shape), ret, ivy.default(ivy.default(out, x), ivy.zeros_like(ret)), out=out)
    return ret


def heaviside(
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
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    if type(x2) != ivy.Array:
        x2 = x1.full_like(x2)
    ret = ivy.where(ivy.equal(x, x1.full_like(0.0)), x2, x1.full_like(0.0), out=out)
    if where is not None:
        ret = ivy.where(ivy.broadcast_to(where, x.shape), ret, ivy.default(ivy.default(out, x), ivy.zeros_like(ret)), out=out)
    return ret


def nan_to_num(
    x, 
    copy=True, 
    nan=0.0, 
    posinf=None, 
    neginf=None
):
    out = x if copy else None
    bounds = ivy.finfo(x) if ivy.is_float_dtype(x) else ivy.iinfo(x)
    pinf = posinf if posinf is not None else bounds.max
    ninf = neginf if neginf is not None else bounds.min
    ret = ivy.where(ivy.equal(x, x.full_like(math.nan)), x.full_like(nan), x, out = out)
    ret = ivy.where(ivy.equal(x, x.full_like(math.inf)), x.full_like(pinf), x, out = out)
    ret = ivy.where(ivy.equal(x, x.full_like(-math.inf)), x.full_like(ninf), x, out = out)
    return ret


def real_if_close(x, tol=100):
    return x # ivy doesn't yet support complex numbers


def interp(x, xp, fp, left=None, right=None, period=None):
    assert(xp.ndim == 1 and fp.ndim == 1)
    assert(xp.shape[0] == fp.shape[0])
    if period is not None:
        assert(period != 0)
        period = ivy.abs(period)
        left = None
        right = None
        xp = ivy.remainer(xp, period)
        yp = ivy.remainer(xp, period)
        asort_xp = ivy.argsort(xp)
        xp = xp[asort_xp]
        yp = yp[asort_xp]
        xp = ivy.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = ivy.concatenate((fp[-1:], fp, fp[0:1]))

    lower = None
    for i in range(0, xp.size[0]-1):
        if x == xp[i]:
            return yp[i]
        elif x == xp[i+1]:
            return yp[i+1]
        elif x > xp[1] and x < xp[i+1]:
            lower = i
            break

    assert(lower is not None)
    
    dist = (x-xp[i]) / (xp[i+1] - xp[i])
    return (fp[i+1] - fp[i])*dist