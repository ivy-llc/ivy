import warnings

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_frontend_arrays,
)
from ivy.functional.frontends.jax.numpy import promote_types_of_jax_inputs


@to_ivy_arrays_and_back
def absolute(x):
    return ivy.abs(x)


abs = absolute


@to_ivy_arrays_and_back
def add(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
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
def argmax(a, axis=None, out=None, keepdims=False):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


def argwhere(a, /, *, size=None, fill_value=None):
    if size is None and fill_value is None:
        return ivy.argwhere(a)

    result = ivy.matrix_transpose(
        ivy.vstack(ivy.nonzero(a, size=size, fill_value=fill_value))
    )
    num_of_dimensions = a.ndim

    if num_of_dimensions == 0:
        return result[:0].reshape(result.shape[0], 0)

    return result.reshape(result.shape[0], num_of_dimensions)


@to_ivy_arrays_and_back
def argsort(a, axis=-1, kind="stable", order=None):
    if kind != "stable":
        warnings.warn(
            "'kind' argument to argsort is ignored; only 'stable' sorts "
            "are supported."
        )
    if order is not None:
        raise ivy.exceptions.IvyError("'order' argument to argsort is not supported.")

    return ivy.argsort(a, axis=axis)


@to_ivy_arrays_and_back
def asarray(
    a,
    dtype=None,
    order=None,
):
    return ivy.asarray(a, dtype=dtype)


@to_ivy_arrays_and_back
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return ivy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


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
        a, a_max = promote_types_of_jax_inputs(a, a_max)
        return ivy.minimum(a, a_max, out=out)
    if a_max is None:
        a, a_min = promote_types_of_jax_inputs(a, a_min)
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
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.matmul(a, b)


@to_ivy_arrays_and_back
def einsum(
    subscripts,
    *operands,
    out=None,
    optimize="optimal",
    precision=None,
    _use_xeinsum=False,
):
    return ivy.einsum(subscripts, *operands, out=out)


@to_ivy_arrays_and_back
def floor(x):
    return ivy.floor(x)


@to_ivy_arrays_and_back
def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    ret = ivy.mean(a, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ivy.astype(ret, ivy.as_ivy_dtype(dtype), copy=False)


@to_ivy_arrays_and_back
def mod(x1, x2):
    return ivy.remainder(x1, x2)


@to_ivy_arrays_and_back
def reshape(a, newshape, order="C"):
    return ivy.reshape(a, shape=newshape, order=order)


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


@to_ivy_arrays_and_back
def uint16(x):
    return ivy.astype(x, ivy.UintDtype("uint16"), copy=False)


@to_ivy_arrays_and_back
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    ret = ivy.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ivy.astype(ret, ivy.as_ivy_dtype(dtype), copy=False)


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
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.pow(x1, x2)


@outputs_to_frontend_arrays
def arange(start, stop=None, step=None, dtype=None):
    return ivy.arange(start, stop, step=step, dtype=dtype)


@to_ivy_arrays_and_back
def bincount(x, weights=None, minlength=0, *, length=None):
    x_list = []
    for i in range(x.shape[0]):
        x_list.append(int(x[i]))
    max_val = int(ivy.max(ivy.array(x_list)))
    ret = [x_list.count(i) for i in range(0, max_val + 1)]
    ret = ivy.array(ret)
    ret = ivy.astype(ret, ivy.as_ivy_dtype(ivy.int64))
    return ret


@to_ivy_arrays_and_back
def cumprod(a, axis=0, dtype=None, out=None):
    if dtype is None:
        dtype = ivy.as_ivy_dtype(a.dtype)
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
def sum(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=None,
    promote_integers=True,
):
    if initial:
        s = ivy.shape(a)
        axis = -1
        header = ivy.full(s, initial)
        a = ivy.concat([a, header], axis=axis)

    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a.dtype) else ivy.as_ivy_dtype(a.dtype)

    # TODO: promote_integers is only supported from JAX v0.3.14
    if dtype is None and promote_integers:
        if ivy.is_bool_dtype(dtype):
            dtype = ivy.default_int_dtype()
        elif ivy.is_uint_dtype(dtype):
            if ivy.dtype_bits(dtype) < ivy.dtype_bits(ivy.default_uint_dtype()):
                dtype = ivy.default_uint_dtype()
        elif ivy.is_int_dtype(dtype):
            if ivy.dtype_bits(dtype) < ivy.dtype_bits(ivy.default_int_dtype()):
                dtype = ivy.default_int_dtype()

    ret = ivy.sum(a, axis=axis, keepdims=keepdims, out=out)

    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ivy.astype(ret, ivy.as_ivy_dtype(dtype))


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
def any(a, axis=None, out=None, keepdims=False, *, where=None):
    # TODO: Out not supported
    ret = ivy.any(a, axis=axis, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(None, ivy.zeros_like(ret)))
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


@to_ivy_arrays_and_back
def hstack(tup, dtype=None):
    # TODO: dtype supported in JAX v0.3.20
    return ivy.hstack(tup)


@to_ivy_arrays_and_back
def arctanh(x):
    return ivy.atanh(x)


@to_ivy_arrays_and_back
def maximum(x1, x2):
    return ivy.maximum(x1, x2)


@to_ivy_arrays_and_back
def minimum(x1, x2):
    return ivy.minimum(x1, x2)


@to_ivy_arrays_and_back
def msort(a):
    return ivy.msort(a)


@to_ivy_arrays_and_back
def multiply(x1, x2):
    return ivy.multiply(x1, x2)


alltrue = all

sometrue = any


@to_ivy_arrays_and_back
def not_equal(x1, x2):
    return ivy.not_equal(x1, x2)


@to_ivy_arrays_and_back
def less(x1, x2):
    return ivy.less(x1, x2)


@to_ivy_arrays_and_back
def less_equal(x1, x2):
    return ivy.less_equal(x1, x2)


@to_ivy_arrays_and_back
def greater(x1, x2):
    return ivy.greater(x1, x2)


@to_ivy_arrays_and_back
def greater_equal(x1, x2):
    return ivy.greater_equal(x1, x2)


@to_ivy_arrays_and_back
def equal(x1, x2):
    return ivy.equal(x1, x2)


@to_ivy_arrays_and_back
def min(a, axis=None, out=None, keepdims=False, where=None):
    ret = ivy.min(a, axis=axis, out=out, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


amin = min


@to_ivy_arrays_and_back
def max(a, axis=None, out=None, keepdims=False, where=None):
    ret = ivy.max(a, axis=axis, out=out, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


amax = max


@to_ivy_arrays_and_back
def log10(x):
    return ivy.log10(x)


@to_ivy_arrays_and_back
def logaddexp(x1, x2):
    return ivy.logaddexp(x1, x2)


@to_ivy_arrays_and_back
def diagonal(a, offset=0, axis1=0, axis2=1):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@to_ivy_arrays_and_back
def expand_dims(a, axis):
    return ivy.expand_dims(a, axis=axis)


@to_ivy_arrays_and_back
def degrees(x):
    return ivy.rad2deg(x)


def eye(N, M=None, k=0, dtype=None):
    return ivy.eye(N, M, k=k, dtype=dtype)


@to_ivy_arrays_and_back
def stack(arrays, axis=0, out=None, dtype=None):
    if dtype:
        return ivy.astype(
            ivy.stack(arrays, axis=axis, out=out), ivy.as_ivy_dtype(dtype)
        )
    return ivy.stack(arrays, axis=axis, out=out)


@to_ivy_arrays_and_back
def take(
    a,
    indices,
    axis=None,
    out=None,
    mode=None,
    unique_indices=False,
    indices_are_sorted=False,
    fill_value=None,
):
    return ivy.take_along_axis(a, indices, axis, out=out)


@to_ivy_arrays_and_back
def zeros_like(a, dtype=None, shape=None):
    if shape:
        return ivy.zeros(shape, dtype=dtype)
    return ivy.zeros_like(a, dtype=dtype)


@to_ivy_arrays_and_back
def negative(
    x,
    /,
):
    return ivy.negative(x)


@to_ivy_arrays_and_back
def rad2deg(
    x, 
    /,
):
    return ivy.rad2deg(x)


@to_ivy_arrays_and_back
def tensordot(a, b, axes=2):
    return ivy.tensordot(a, b, axes=axes)
