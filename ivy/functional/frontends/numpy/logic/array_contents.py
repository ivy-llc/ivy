# global
import ivy


def isneginf(x, out=None):
    isinf = ivy.isinf(x)
    neg_sign_bit = ivy.less(x, 0)
    return ivy.logical_and(isinf, neg_sign_bit, out=out)


def isposinf(x, out=None):
    isinf = ivy.isinf(x)
    pos_sign_bit = ivy.bitwise_invert(ivy.less(x, 0))
    return ivy.logical_and(isinf, pos_sign_bit, out=out)


def _compute_allclose_with_tol(a, b, rtol, atol):
    return ivy.all(
        ivy.less_equal(
            ivy.abs(ivy.subtract(a, b)),
            ivy.add(atol, ivy.multiply(rtol, ivy.abs(b))),
        )
    )


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    finite_a = ivy.isfinite(a)
    finite_b = ivy.isfinite(b)

    if ivy.all(finite_a) and ivy.all(finite_b):
        ret = _compute_allclose_with_tol(a, b, rtol, atol)
        return ivy.array(ivy.all_equal(True, ret))
    else:
        finites = ivy.bitwise_and(finite_a, finite_b)
        ret = ivy.zeros_like(finites)
        ret_ = ret.astype(int)
        a = a * ivy.ones_like(ret_)
        b = b * ivy.ones_like(ret_)
        ret[finites] = _compute_allclose_with_tol(
            a[finites], b[finites], rtol, atol
        )
        nans = ivy.bitwise_invert(finites)
        ret[nans] = ivy.equal(a[nans], b[nans])
        if equal_nan:
            both_nan = ivy.bitwise_and(ivy.isnan(a), ivy.isnan(b))
            ret[both_nan] = both_nan[both_nan]
        return ivy.all(ret)


def _compute_isclose_with_tol(a, b, rtol, atol):
    return ivy.less_equal(
        ivy.abs(ivy.subtract(a, b)),
        ivy.add(atol, ivy.multiply(rtol, ivy.abs(b))),
    )


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    finite_a = ivy.isfinite(a)
    finite_b = ivy.isfinite(b)
    if ivy.all(finite_a) and ivy.all(finite_b):
        return _compute_isclose_with_tol(a, b, rtol, atol)

    else:
        finites = ivy.bitwise_and(finite_a, finite_b)
        ret = ivy.zeros_like(finites)
        ret_ = ret.astype(int)
        a = a * ivy.ones_like(ret_)
        b = b * ivy.ones_like(ret_)
        ret[finites] = _compute_isclose_with_tol(
            a[finites], b[finites], rtol, atol
        )
        nans = ivy.bitwise_invert(finites)
        ret[nans] = ivy.equal(a[nans], b[nans])
        if equal_nan:
            both_nan = ivy.bitwise_and(ivy.isnan(a), ivy.isnan(b))
            ret[both_nan] = both_nan[both_nan]
        return ret
