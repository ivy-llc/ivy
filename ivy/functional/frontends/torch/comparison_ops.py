# global
import ivy


def _compute_close_with_tol(input, other, rtol, atol):
    in_sub = ivy.abs(ivy.subtract(input, other))
    tol_sum = rtol + atol
    other_abs = ivy.abs(other)
    prod = ivy.multiply(tol_sum, other_abs)
    comp = ivy.less_equal(in_sub, prod)
    return ivy.all(comp)


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    finite_input = ivy.isfinite(input)
    finite_other = ivy.isfinite(other)
    if ivy.all(finite_input) and ivy.all(finite_other):
        _compute_close_with_tol(input, other, rtol, atol)
    else:
        finites = ivy.bitwise_and(finite_input, finite_other)
        ret = ivy.zeros_like(finites)
        ret_ = ret.astype(int)
        input = input * ivy.ones_like(ret_)
        other = other * ivy.ones_like(ret_)
        ret[finites] = _compute_close_with_tol(
            input[finites], other[finites], rtol, atol
        )
        nans = ivy.bitwise_invert(finites)
        ret[nans] = ivy.equal(input[nans], other[nans])
        if equal_nan:
            both_nan = ivy.bitwise_and(ivy.isnan(input), ivy.isnan(other))
            ret[both_nan] = both_nan[both_nan]
        return ret


allclose.unsupported_dtypes = ("float16",)
