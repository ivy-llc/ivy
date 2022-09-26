# local
import ivy


def abs(x):
    return ivy.abs(x)


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
