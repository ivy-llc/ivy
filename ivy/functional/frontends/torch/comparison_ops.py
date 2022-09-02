# ToDo: Add allclose(), isclose(), isposinf(), isneginf(), fmax() to functional API
# global
import ivy

# local
from collections import namedtuple


def _compute_allclose_with_tol(input, other, rtol, atol):
    ret = ivy.less_equal(
        ivy.abs(ivy.subtract(input, other)),
        ivy.add(atol, ivy.multiply(rtol, ivy.abs(other))),
    )
    return ivy.all(ret)


def _compute_isclose_with_tol(input, other, rtol, atol):
    return ivy.less_equal(
        ivy.abs(ivy.subtract(input, other)),
        ivy.add(atol, ivy.multiply(rtol, ivy.abs(other))),
    )


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    finite_input = ivy.isfinite(input)
    finite_other = ivy.isfinite(other)
    if ivy.all(finite_input) and ivy.all(finite_other):
        ret = _compute_allclose_with_tol(input, other, rtol, atol)
        return bool(ret)
    else:
        finites = ivy.bitwise_and(finite_input, finite_other)
        ret = ivy.zeros_like(finites)
        ret_ = ret.astype(int)
        input = input * ivy.ones_like(ret_)
        other = other * ivy.ones_like(ret_)
        ret[finites] = _compute_allclose_with_tol(
            input[finites], other[finites], rtol, atol
        )
        nans = ivy.bitwise_invert(finites)
        ret[nans] = ivy.equal(input[nans], other[nans])
        if equal_nan:
            both_nan = ivy.bitwise_and(ivy.isnan(input), ivy.isnan(other))
            ret[both_nan] = both_nan[both_nan]
        return ivy.all(ret)


allclose.unsupported_dtypes = ("float16",)


def equal(input, other):
    ret = ivy.all_equal(input, other, equality_matrix=False)
    return bool(ret)


def eq(input, other, *, out=None):
    ret = ivy.equal(input, other, out=out)
    return ret


def argsort(input, dim=-1, descending=False):
    return ivy.argsort(input, axis=dim, descending=descending)


def greater_equal(input, other, *, out=None):
    ret = ivy.greater_equal(input, other, out=out)
    return ret


ge = greater_equal


def greater(input, other, *, out=None):
    return ivy.greater(input, other, out=out)


gt = greater


def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    finite_input = ivy.isfinite(input)
    finite_other = ivy.isfinite(other)
    if ivy.all(finite_input) and ivy.all(finite_other):
        return _compute_isclose_with_tol(input, other, rtol, atol)

    else:
        finites = ivy.bitwise_and(finite_input, finite_other)
        ret = ivy.zeros_like(finites)
        ret_ = ret.astype(int)
        input = input * ivy.ones_like(ret_)
        other = other * ivy.ones_like(ret_)
        ret[finites] = _compute_isclose_with_tol(
            input[finites], other[finites], rtol, atol
        )
        nans = ivy.bitwise_invert(finites)
        ret[nans] = ivy.equal(input[nans], other[nans])
        if equal_nan:
            both_nan = ivy.bitwise_and(ivy.isnan(input), ivy.isnan(other))
            ret[both_nan] = both_nan[both_nan]
        return ret


isclose.unsupported_dtypes = ("float16",)


def isfinite(input):
    return ivy.isfinite(input)


def isinf(input):
    return ivy.isinf(input)


def isposinf(input, *, out=None):
    is_inf = ivy.isinf(input)
    pos_sign_bit = ivy.bitwise_invert(ivy.less(input, 0))
    return ivy.logical_and(is_inf, pos_sign_bit, out=out)


def isneginf(input, *, out=None):
    is_inf = ivy.isinf(input)
    neg_sign_bit = ivy.less(input, 0)
    return ivy.logical_and(is_inf, neg_sign_bit, out=out)


def sort(input, dim=-1, descending=False, stable=False, out=None):
    values = ivy.sort(input, axis=dim, descending=descending, stable=stable, out=out)

    indices = ivy.argsort(input, axis=dim, descending=descending)

    ret = namedtuple("sort", ["values", "indices"])(values, indices)

    return ret


def isnan(input):
    return ivy.isnan(input)


def less_equal(input, other, *, out=None):
    return ivy.less_equal(input, other, out=out)


le = less_equal


def less(input, other, *, out=None):
    return ivy.less(input, other, out=out)


lt = less


def not_equal(input, other, *, out=None):
    return ivy.not_equal(input, other, out=out)


ne = not_equal


def minimum(input, other, *, out=None):
    return ivy.minimum(input, other, out=out)


def fmax(input, other, *, out=None):
    return ivy.where(
        ivy.bitwise_or(ivy.greater(input, other), ivy.isnan(other)),
        input,
        other,
        out=out,
    )
