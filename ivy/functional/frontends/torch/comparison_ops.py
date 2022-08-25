# ToDo: Add allclose() to functional API
# global
import ivy


def _compute_allclose_with_tol(input, other, rtol, atol):
    ret = ivy.less_equal(
        ivy.abs(ivy.subtract(input, other)),
        ivy.add(atol, ivy.multiply(rtol, ivy.abs(other))),
    )
    return ivy.all(ret)


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
