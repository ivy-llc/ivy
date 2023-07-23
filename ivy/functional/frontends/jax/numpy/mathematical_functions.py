# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.jax.numpy import promote_types_of_jax_inputs
from ivy.functional.frontends.numpy.manipulation_routines import trim_zeros


# sign
@to_ivy_arrays_and_back
def sign(x, /):
    return ivy.sign(x, out=None)


@to_ivy_arrays_and_back
def absolute(x, /):
    return ivy.abs(x)


abs = absolute


@to_ivy_arrays_and_back
def add(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.add(x1, x2)


@to_ivy_arrays_and_back
def imag(val, /):
    return ivy.imag(val)


@to_ivy_arrays_and_back
def angle(z, deg=False):
    return ivy.angle(z, deg=deg)


@to_ivy_arrays_and_back
def diff(a, n=1, axis=-1, prepend=None, append=None):
    return ivy.diff(a, n=n, axis=axis, prepend=prepend, append=append, out=None)


@to_ivy_arrays_and_back
def ediff1d(ary, to_end=None, to_begin=None):
    diffs = ivy.diff(ary)
    diffs_dtype = diffs.dtype
    if to_begin is not None:
        if not isinstance(to_begin, (list, tuple)):
            to_begin = [to_begin]
        to_begin = ivy.array(to_begin, dtype=diffs_dtype)
        diffs = ivy.concat((to_begin, diffs))
    if to_end is not None:
        if not isinstance(to_end, (list, tuple)):
            to_end = [to_end]
        to_end = ivy.array(to_end, dtype=diffs_dtype)
        diffs = ivy.concat((diffs, to_end))
    return diffs


@to_ivy_arrays_and_back
def arctan(x, /):
    return ivy.atan(x)


@to_ivy_arrays_and_back
def arctan2(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.atan2(x1, x2)


@to_ivy_arrays_and_back
def convolve(a, v, mode="full", *, precision=None):
    a, v = promote_types_of_jax_inputs(a, v)

    if len(a) < len(v):
        a, v = v, a
    v = ivy.flip(v)

    out_order = slice(None)

    if mode == "valid":
        padding = [(0, 0)]
    elif mode == "same":
        padding = [(v.shape[0] // 2, v.shape[0] - v.shape[0] // 2 - 1)]
    elif mode == "full":
        padding = [(v.shape[0] - 1, v.shape[0] - 1)]

    a = a.reshape([1, 1, a.shape[0]])
    v = v.reshape([v.shape[0], 1, 1])

    result = ivy.conv_general_dilated(
        a,
        v,
        (1,),
        padding,
        dims=1,
        data_format="channel_first",
    )
    return result[0, 0, out_order]


@to_ivy_arrays_and_back
def cos(x, /):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def cosh(x, /):
    return ivy.cosh(x)


@to_ivy_arrays_and_back
def dot(a, b, *, precision=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.matmul(a, b)


@to_ivy_arrays_and_back
def floor(x, /):
    return ivy.floor(x)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"0.4.13 and below": ("complex",)}, "jax")
def mod(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.remainder(x1, x2)


@to_ivy_arrays_and_back
def modf(x, /, out=None):
    y1 = ivy.where(x >= 0, ivy.floor(x), ivy.ceil(x))  # integral part
    y2 = x - y1  # fractional part
    dtype_str = str(x.dtype)
    if "float" in dtype_str:
        return y2, y1
    # floats return as they were. u/ints (8, 16, 32) return as float32, 64 as float64.
    dtype_size = x.itemsize * 8
    if "int8" in dtype_str or "int16" in dtype_str:
        dtype_size = 32
    ret_type = "float{}".format(dtype_size)
    return y2.astype(ret_type), y1.astype(ret_type)


@to_ivy_arrays_and_back
def divmod(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return tuple([ivy.floor_divide(x1, x2), ivy.remainder(x1, x2)])


@to_ivy_arrays_and_back
def sinh(x, /):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def sin(x, /):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def tan(x, /):
    return ivy.tan(x)


@to_ivy_arrays_and_back
def tanh(x, /):
    return ivy.tanh(x)


@to_ivy_arrays_and_back
def arccos(x, /):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def arccosh(x, /):
    return ivy.acosh(x)


@to_ivy_arrays_and_back
def arcsin(x, /):
    return ivy.asin(x)


@to_ivy_arrays_and_back
def arcsinh(x, /):
    return ivy.asinh(x)


@to_ivy_arrays_and_back
def power(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.pow(x1, x2)


@to_ivy_arrays_and_back
def trunc(x):
    return ivy.trunc(x)


@to_ivy_arrays_and_back
def ceil(x, /):
    return ivy.ceil(x)


@to_ivy_arrays_and_back
def float_power(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.float_power(x1, x2).astype(x1.dtype, copy=False)


@to_ivy_arrays_and_back
def deg2rad(x, /):
    return ivy.deg2rad(x)


@to_ivy_arrays_and_back
def radians(x, /):
    return ivy.deg2rad(x)


@to_ivy_arrays_and_back
def exp2(x, /):
    return ivy.exp2(x)


@to_ivy_arrays_and_back
def gcd(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.gcd(x1, x2)


@to_ivy_arrays_and_back
def i0(x):
    return ivy.i0(x)


@to_ivy_arrays_and_back
def kron(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.kron(a, b)


@to_ivy_arrays_and_back
def lcm(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.lcm(x1, x2)


@to_ivy_arrays_and_back
def logaddexp2(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.logaddexp2(x1, x2)


@to_ivy_arrays_and_back
def trapz(y, x=None, dx=1.0, axis=-1, out=None):
    return ivy.trapz(y, x=x, dx=dx, axis=axis, out=out)


@to_ivy_arrays_and_back
def sqrt(x, /):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def square(x, /):
    return ivy.square(x)


@to_ivy_arrays_and_back
def arctanh(x, /):
    return ivy.atanh(x)


@to_ivy_arrays_and_back
def multiply(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.multiply(x1, x2)


@to_ivy_arrays_and_back
def matmul(a, b, *, precision=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.matmul(a, b)


@to_ivy_arrays_and_back
def log10(x, /):
    return ivy.log10(x)


@to_ivy_arrays_and_back
def logaddexp(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.logaddexp(x1, x2)


@to_ivy_arrays_and_back
def degrees(x, /):
    return ivy.rad2deg(x)


@to_ivy_arrays_and_back
def negative(
    x,
    /,
):
    return ivy.negative(x)


@to_ivy_arrays_and_back
def positive(
    x,
    /,
):
    return ivy.positive(x)


@to_ivy_arrays_and_back
def rad2deg(
    x,
    /,
):
    return ivy.rad2deg(x)


@to_ivy_arrays_and_back
def tensordot(a, b, axes=2):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.tensordot(a, b, axes=axes)


@to_ivy_arrays_and_back
def divide(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    if ivy.dtype(x1) in ["int64", "uint64"]:
        x1 = ivy.astype(x1, ivy.float64)
    elif ivy.is_int_dtype(x1):
        x1 = ivy.astype(x1, ivy.float32)

    return ivy.divide(x1, x2).astype(x1.dtype)


true_divide = divide


@to_ivy_arrays_and_back
def exp(
    x,
    /,
):
    return ivy.exp(x)


@to_ivy_arrays_and_back
def expm1(
    x,
    /,
):
    return ivy.expm1(x)


@to_ivy_arrays_and_back
def fmax(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    ret = ivy.where(
        ivy.bitwise_or(ivy.greater(x1, x2), ivy.isnan(x2)),
        x1,
        x2,
    )
    return ret


@to_ivy_arrays_and_back
def fmin(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    ret = ivy.where(
        ivy.bitwise_or(ivy.less(x1, x2), ivy.isnan(x2)),
        x1,
        x2,
    )
    print("jax-frontend", ret)
    return ret


@with_unsupported_dtypes(
    {"0.4.13 and below": ("uint16",)},
    "jax",
)
@to_ivy_arrays_and_back
def fabs(x, /):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def fmod(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.fmod(x1, x2)


@to_ivy_arrays_and_back
def maximum(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.maximum(x1, x2)


@to_ivy_arrays_and_back
def minimum(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.minimum(x1, x2)


@to_ivy_arrays_and_back
def heaviside(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.heaviside(x1, x2)


@to_ivy_arrays_and_back
def log(x, /):
    return ivy.log(x)


@to_ivy_arrays_and_back
def log1p(x, /):
    return ivy.log1p(x)


@to_ivy_arrays_and_back
def copysign(x1, x2, /):
    return ivy.copysign(x1, x2)


@to_ivy_arrays_and_back
def sinc(x, /):
    return ivy.sinc(x)


@with_unsupported_dtypes(
    {
        "0.4.13 and below": (
            "bfloat16",
            "float16",
        )
    },
    "jax",
)
@to_ivy_arrays_and_back
def nextafter(x1, x2, /):
    return ivy.nextafter(x1, x2)


@to_ivy_arrays_and_back
def remainder(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.remainder(x1, x2)


@to_ivy_arrays_and_back
def trace(a, offset=0, axis1=0, axis2=1, out=None):
    return ivy.trace(a, offset=offset, axis1=axis1, axis2=axis2, out=out)


@to_ivy_arrays_and_back
def log2(x, /):
    return ivy.log2(x)


@to_ivy_arrays_and_back
def vdot(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.multiply(a, b).sum()


@with_unsupported_dtypes(
    {"0.4.13 and below": ("bfloat16",)},
    "jax",
)
@to_ivy_arrays_and_back
def cbrt(x, /):
    all_positive = ivy.pow(ivy.abs(x), 1.0 / 3.0)
    return ivy.where(ivy.less(x, 0.0), ivy.negative(all_positive), all_positive)


@to_ivy_arrays_and_back
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return ivy.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


@to_ivy_arrays_and_back
def fix(x, out=None):
    return ivy.fix(x, out=out)


@to_ivy_arrays_and_back
def real(val, /):
    return ivy.real(val)


@to_ivy_arrays_and_back
def hypot(x1, x2, /):
    return ivy.hypot(x1, x2)


@to_ivy_arrays_and_back
def floor_divide(x1, x2, /, out=None):
    return ivy.floor_divide(x1, x2, out=out)


@to_ivy_arrays_and_back
def inner(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.inner(a, b)


@to_ivy_arrays_and_back
def outer(a, b, out=None):
    return ivy.outer(a, b, out=out)


@to_ivy_arrays_and_back
def reciprocal(x, /):
    return ivy.reciprocal(x)


@to_ivy_arrays_and_back
def conj(x, /):
    return ivy.conj(x)


@to_ivy_arrays_and_back
def subtract(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.subtract(x1, x2)


@to_ivy_arrays_and_back
def around(a, decimals=0, out=None):
    ret_dtype = a.dtype
    return ivy.round(a, decimals=decimals, out=out).astype(ret_dtype, copy=False)


@to_ivy_arrays_and_back
def round(a, decimals=0, out=None):
    return ivy.round(a, decimals=decimals, out=out)


@to_ivy_arrays_and_back
def frexp(x, /):
    return ivy.frexp(x)


@to_ivy_arrays_and_back
def ldexp(x1, x2, /):
    return ivy.ldexp(x1, x2)


@to_ivy_arrays_and_back
def poly(seq_of_zeros):
    seq_of_zeros = ivy.atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape
    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq_of_zeros = ivy.eigvals(seq_of_zeros)
    if seq_of_zeros.ndim != 1:
        raise ValueError("input must be 1d or non-empty square 2d array.")
    dt = seq_of_zeros.dtype
    if len(seq_of_zeros) == 0:
        return ivy.ones((), dtype=dt)
    a = ivy.ones((1,), dtype=dt)
    for k in range(len(seq_of_zeros)):
        a = convolve(
            a, ivy.asarray([ivy.array(1), -seq_of_zeros[k]], dtype=dt), mode="full"
        )
    return a


@to_ivy_arrays_and_back
def polyadd(a1, a2):
    d = max(a1.size, a2.size)
    a1 = ivy.pad(a1, (d - a1.size, 0), mode="constant")
    a2 = ivy.pad(a2, (d - a2.size, 0), mode="constant")
    return a1 + a2


@with_unsupported_dtypes(
    {"0.4.13 and below": ("float16",)},
    "jax",
)
@to_ivy_arrays_and_back
def polyder(p, m=1):
    if m < 0:
        raise ValueError("Order of derivative must be positive.")

    if m == 0:
        return p
    p_dtype = p.dtype
    coeff = ivy.prod(
        ivy.expand_dims(ivy.arange(m, len(p), dtype=p_dtype))
        - ivy.expand_dims(ivy.arange(m, dtype=p_dtype), axis=1),
        axis=0,
    )
    return (p[:-m] * coeff[::-1]).astype(p_dtype)


@with_unsupported_dtypes(
    {"0.4.13 and below": ("float16",)},
    "jax",
)
@to_ivy_arrays_and_back
def polyint(p, m=1, k=None):
    p = ivy.asarray(p)
    m = int(m)
    if m == 0:
        return p
    if k is None:
        k_arr = ivy.zeros((m,), dtype=p.dtype)
    elif isinstance(k, (int, float)):
        k_arr = ivy.full((m,), k, dtype=p.dtype)
    elif ivy.asarray(k).shape == (1,):
        k_arr = ivy.full((m,), ivy.asarray(k)[0], dtype=p.dtype)
    elif ivy.asarray(k).shape == (m,):
        k_arr = ivy.asarray(k, dtype=p.dtype)
    else:
        raise ValueError("k must be a scalar or a rank-1 array of length 1 or m.")
    grid = (
        ivy.arange(p.size + m, dtype=p.dtype)[ivy.newaxis]
        - ivy.arange(m, dtype=p.dtype)[:, ivy.newaxis]
    )
    coeff = ivy.maximum(1, grid).prod(axis=0)[::-1]
    return ivy.divide(ivy.concat((p, k_arr)), coeff).astype(p.dtype)


@with_unsupported_dtypes(
    {"0.3.14 and below": ("float16",)},
    "jax",
)
@to_ivy_arrays_and_back
def polydiv(u, v, *, trim_leading_zeros=False):
    u, v_arr = ivy.promote_types_of_inputs(u, v)
    n = v_arr.shape[0] - 1
    m = u.shape[0] - 1
    scale = 1.0 / v_arr[0]
    q = ivy.zeros((max(m - n + 1, 1),), dtype=u.dtype)
    r = ivy.copy_array(u)
    for k in range(0, m - n + 1):
        d = scale * r[k]
        q[k] = d
        r[k : k + n + 1] = r[k : k + n + 1] - (d * v_arr)
    # if trim_leading_zeros:
    #    r = trim_zeros_tol(r, trim='f')
    # TODO: need to control tolerance of this function to handle the argument
    return q, r


@to_ivy_arrays_and_back
def polysub(a1, a2):
    n = max(a1.size, a2.size) - 1
    a1 = ivy.pad(a1, (0, n - a1.size + 1), mode="constant")
    a2 = ivy.pad(a2, (0, n - a2.size + 1), mode="constant")
    return a1 - a2


@to_ivy_arrays_and_back
def polymul(a1, a2, *, trim_leading_zeros=False):
    a1, a2 = ivy.atleast_1d(a1), ivy.atleast_1d(a2)
    if trim_leading_zeros and (len(a1) > 1 or len(a2) > 1):
        a1, a2 = trim_zeros(a1, trim="f"), trim_zeros(a2, trim="f")
    if len(a1) == 0:
        a1 = ivy.asarray([0], dtype=a1.dtype)
    if len(a2) == 0:
        a2 = ivy.asarray([0], dtype=a2.dtype)
    return convolve(a1, a2, mode="full")


@to_ivy_arrays_and_back
def signbit(x, /):
    x = ivy.array(x)
    return ivy.signbit(x)


@to_ivy_arrays_and_back
def product(
    a,
    *,
    axis=None,
    dtype=None,
    keepdims=False,
    initial=None,
    where=None,
    promote_integers=True,
    out=None,
):
    if ivy.is_array(where):
        a = ivy.where(where, a, ivy.default(out, ivy.ones_like(a)), out=out)
    if promote_integers:
        if dtype is None:
            dtype = a.dtype
    if initial is not None:
        if axis is not None:
            s = ivy.to_list(ivy.shape(a, as_array=True))
            s[axis] = 1
            header = ivy.full(ivy.Shape(tuple(s)), initial)
            a = ivy.concat([header, a], axis=axis)
        else:
            a[0] *= initial
    return ivy.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def conjugate(x, /):
    return ivy.conj(x)

@to_ivy_arrays_and_back
def cos(x):
    return ivy.cos(x)
    
