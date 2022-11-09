# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def abs(x):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def absolute(x):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def add(x1, x2):
    x1, x2 = ivy.frontends.jax.promote_types_of_jax_inputs(x1, x2)
    return ivy.add(x1, x2)


@to_ivy_arrays_and_back
def all(a, axis=None, out=None, keepdims=False, *, where=False):
    return ivy.all(a, axis=axis, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def arctan(x):
    ret = ivy.atan(x)
    return ret


@to_ivy_arrays_and_back
def arctan2(x1, x2):
    return ivy.atan2(x1, x2)


@to_ivy_arrays_and_back
def argmax(a, axis=None, out=None, keepdims=None):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


def _compute_allclose_with_tol(input, other, rtol, atol):
    return ivy.all(
        ivy.less_equal(
            ivy.abs(ivy.subtract(input, other)),
            ivy.add(atol, ivy.multiply(rtol, ivy.abs(other))),
        )
    )


def _compute_isclose_with_tol(input, other, rtol, atol):
    return ivy.less_equal(
        ivy.abs(ivy.subtract(input, other)),
        ivy.add(atol, ivy.multiply(rtol, ivy.abs(other))),
    )


@to_ivy_arrays_and_back
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    finite_input = ivy.isfinite(a)
    finite_other = ivy.isfinite(b)
    if ivy.all(finite_input) and ivy.all(finite_other):
        ret = _compute_allclose_with_tol(a, b, rtol, atol)
        ret = ivy.all_equal(True, ret)
    else:
        finites = ivy.bitwise_and(finite_input, finite_other)
        ret = ivy.zeros_like(finites)
        ret_ = ret.astype(int)
        input = a * ivy.ones_like(ret_)
        other = b * ivy.ones_like(ret_)
        ret[finites] = _compute_allclose_with_tol(
            input[finites], other[finites], rtol, atol
        )
        nans = ivy.bitwise_invert(finites)
        ret[nans] = ivy.equal(input[nans], other[nans])
        if equal_nan:
            both_nan = ivy.bitwise_and(ivy.isnan(input), ivy.isnan(other))
            ret[both_nan] = both_nan[both_nan]
        ret = ivy.all(ret)
    return ivy.array(ret, dtype=ivy.bool)


@to_ivy_arrays_and_back
def broadcast_to(arr, shape):
    return ivy.broadcast_to(arr, shape)


@to_ivy_arrays_and_back
def clip(a, a_min=None, a_max=None, out=None):
    ivy.assertions.check_all_or_any_fn(
        a_min,
        a_max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of a_min or a_max can be None",
    )
    a = ivy.array(a)
    if a_min is None:
        a, a_max = ivy.frontends.jax.promote_types_of_jax_inputs(a, a_max)
        return ivy.minimum(a, a_max, out=out)
    if a_max is None:
        a, a_min = ivy.frontends.jax.promote_types_of_jax_inputs(a, a_min)
        return ivy.maximum(a, a_min, out=out)
    return ivy.clip(a, a_min, a_max, out=out)


@to_ivy_arrays_and_back
def concatenate(arrays, axis=0, dtype=None):
    ret = ivy.concat(arrays, axis=axis)
    if dtype:
        ret = ivy.array(ret, dtype=dtype)
    return ret


@to_ivy_arrays_and_back
def cos(x):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def cosh(x):
    return ivy.cosh(x)


@to_ivy_arrays_and_back
def dot(a, b, *, precision=None):
    a, b = ivy.frontends.jax.promote_types_of_jax_inputs(a, b)
    return ivy.matmul(a, b)


@to_ivy_arrays_and_back
def einsum(
    subscripts,
    *operands,
    out=None,
    optimize="optimal",
    precision=None,
    _use_xeinsum=False
):
    return ivy.einsum(subscripts, *operands, out=out)


@to_ivy_arrays_and_back
def floor(x):
    return ivy.floor(x)


@to_ivy_arrays_and_back
def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=None):
    a = ivy.array(a)
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    ret = ivy.mean(a, axis=axis, out=out, keepdims=keepdims)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret.astype(dtype, copy=False)


@to_ivy_arrays_and_back
def mod(x1, x2):
    return ivy.remainder(x1, x2)


@to_ivy_arrays_and_back
def reshape(a, newshape, order="C"):
    return ivy.reshape(a, newshape)


@to_ivy_arrays_and_back
def sinh(x):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def sin(x):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def tan(x):
    return ivy.tan(x)


@to_ivy_arrays_and_back
def tanh(x):
    return ivy.tanh(x)


def uint16(x):
    return ivy.astype(x, ivy.uint16)


@to_ivy_arrays_and_back
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None):
    a = ivy.array(a)
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    ret = ivy.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret.astype(dtype, copy=False)


@to_ivy_arrays_and_back
def arccos(x):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def arccosh(x):
    return ivy.acosh(x)


@to_ivy_arrays_and_back
def arcsin(x):
    return ivy.asin(x)


@to_ivy_arrays_and_back
def arcsinh(x):
    return ivy.asinh(x)


@to_ivy_arrays_and_back
def fmax(x1, x2):
    ret = ivy.where(
        ivy.bitwise_or(ivy.greater(x1, x2), ivy.isnan(x2)),
        x1,
        x2,
    )
    return ret


@to_ivy_arrays_and_back
def argmin(a, axis=None, out=None, keepdims=None):
    return ivy.argmin(a, axis=axis, out=out, keepdims=keepdims)


@to_ivy_arrays_and_back
def array_equal(a1, a2, equal_nan: bool) -> bool:
    try:
        a1, a2 = ivy.asarray(a1), ivy.asarray(a2)
    except Exception:
        return False
    if ivy.shape(a1) != ivy.shape(a2):
        return False
    eq = ivy.asarray(a1 == a2)
    if equal_nan:
        eq = ivy.logical_or(eq, ivy.logical_and(ivy.isnan(a1), ivy.isnan(a2)))
    return ivy.all(eq)


@to_ivy_arrays_and_back
def array_equiv(a1, a2) -> bool:
    try:
        a1, a2 = ivy.asarray(a1), ivy.asarray(a2)
    except Exception:
        return False
    try:
        eq = ivy.equal(a1, a2)
    except ValueError:
        # shapes are not broadcastable
        return False
    return ivy.all(eq)


@to_ivy_arrays_and_back
def zeros(shape, dtype=None):
    if dtype is None:
        dtype = ivy.float64
    return ivy.zeros(shape, dtype=dtype)


@to_ivy_arrays_and_back
def bitwise_and(x1, x2):
    return ivy.bitwise_and(x1, x2)


@to_ivy_arrays_and_back
def bitwise_not(x):
    return ivy.bitwise_invert(x)


@to_ivy_arrays_and_back
def bitwise_or(x1, x2):
    return ivy.bitwise_or(x1, x2)


@to_ivy_arrays_and_back
def bitwise_xor(x1, x2):
    return ivy.bitwise_xor(x1, x2)


@to_ivy_arrays_and_back
def moveaxis(a, source, destination):
    return ivy.moveaxis(a, source, destination)


@to_ivy_arrays_and_back
def flipud(m):
    return ivy.flipud(m, out=None)


@to_ivy_arrays_and_back
def power(x1, x2):
    x1, x2 = ivy.frontends.jax.promote_types_of_jax_inputs(x1, x2)
    return ivy.pow(x1, x2)


@to_ivy_arrays_and_back
def arange(start, stop=None, step=None, dtype=None):
    return ivy.arange(start, stop, step=step, dtype=dtype)


@to_ivy_arrays_and_back
def bincount(x, weights=None, minlength=0, *, length=None):
    x_list = []
    for i in range(x.shape[0]):
        x_list.append(int(x[i]))
    ret = [x_list.count(i) for i in range(0, max(x_list) + 1)]
    ret = ivy.array(ret)
    ret = ivy.astype(ret, ivy.as_ivy_dtype(ivy.int64))
    return ret


@to_ivy_arrays_and_back
def cumprod(a, axis=0, dtype=None, out=None):
    if dtype is None:
        dtype = ivy.uint8
    return ivy.cumprod(a, axis, dtype=dtype, out=out)


@to_ivy_arrays_and_back
def trunc(x):
    return ivy.trunc(x)


@to_ivy_arrays_and_back
def ceil(x):
    return ivy.ceil(x)


@to_ivy_arrays_and_back
def float_power(x1, x2):
    return ivy.float_power(x1, x2)


@to_ivy_arrays_and_back
def cumsum(a, axis=0, dtype=None, out=None):
    if dtype is None:
        dtype = ivy.uint8
    return ivy.cumsum(a, axis, dtype=dtype, out=out)


cumproduct = cumprod


@to_ivy_arrays_and_back
def heaviside(x1, x2):
    return ivy.heaviside(x1, x2)


@to_ivy_arrays_and_back
def deg2rad(x):
    return ivy.deg2rad(x)


@to_ivy_arrays_and_back
def exp2(x):
    return ivy.exp2(x)


@to_ivy_arrays_and_back
def gcd(x1, x2):
    return ivy.gcd(x1, x2)


@to_ivy_arrays_and_back
def i0(x):
    return ivy.i0(x)


@to_ivy_arrays_and_back
def isneginf(x, out=None):
    return ivy.isneginf(x, out=out)


@to_ivy_arrays_and_back
def isposinf(x, out=None):
    return ivy.isposinf(x, out=out)


@to_ivy_arrays_and_back
def kron(a, b):
    return ivy.kron(a, b)


@to_ivy_arrays_and_back
def lcm(x1, x2):
    return ivy.lcm(x1, x2)


@to_ivy_arrays_and_back
def logaddexp2(x1, x2):
    return ivy.logaddexp2(x1, x2)


@to_ivy_arrays_and_back
def trapz(y, x=None, dx=1.0, axis=-1, out=None):
    return ivy.trapz(y, x=x, dx=dx, axis=axis, out=out)


@to_ivy_arrays_and_back
def any(a, axis=None, out=None, keepdims=False, *, where=True):
    ret = ivy.any(a, axis=axis, keepdims=keepdims)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)))
    return ret


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)


@to_ivy_arrays_and_back
def flip(m, axis=None):
    return ivy.flip(m, axis=axis)


@to_ivy_arrays_and_back
def fliplr(m):
    return ivy.fliplr(m)
