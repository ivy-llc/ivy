# local
import ivy


def abs(x):
    return ivy.abs(x)


def absolute(x):
    return ivy.abs(x)


def add(x1, x2):
    return ivy.add(x1, x2)


def all(a, axis=None, out=None, keepdims=False, *, where=False):
    return ivy.all(a, axis=axis, keepdims=keepdims, out=out)


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


def broadcast_to(arr, shape):
    return ivy.broadcast_to(arr, shape)


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
        return ivy.minimum(a, a_max, out=out)
    if a_max is None:
        return ivy.maximum(a, a_min, out=out)
    return ivy.clip(a, a_min, a_max, out=out)


# def einsum(*operands, out=None, optimize="optimal", precision=None,
# _use_xeinsum=False):
#     # TODO: optimize, precision handling
#     return ivy.eimsum(equation=optimize, *operands, out=out)


def reshape(a, newshape, order="C"):
    return ivy.reshape(a, newshape)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None):
    a = ivy.array(a)
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    ret = ivy.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret.astype(dtype)
