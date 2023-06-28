# global
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back

# local
from collections import namedtuple


def _compute_allclose_with_tol(input, other, rtol, atol):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.all(
        ivy.less_equal(
            ivy.abs(ivy.subtract(input, other)),
            ivy.add(atol, ivy.multiply(rtol, ivy.abs(other))),
        )
    )


def _compute_isclose_with_tol(input, other, rtol, atol):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.less_equal(
        ivy.abs(ivy.subtract(input, other)),
        ivy.add(atol, ivy.multiply(rtol, ivy.abs(other))),
    )


@to_ivy_arrays_and_back
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    finite_input = ivy.isfinite(input)
    finite_other = ivy.isfinite(other)
    if ivy.all(finite_input) and ivy.all(finite_other):
        ret = _compute_allclose_with_tol(input, other, rtol, atol)
        return ivy.all_equal(True, ret)
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


@to_ivy_arrays_and_back
def equal(input, other):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.all_equal(input, other)


@to_ivy_arrays_and_back
def eq(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.equal(input, other, out=out)


@to_ivy_arrays_and_back
def argsort(input, dim=-1, descending=False):
    return ivy.argsort(input, axis=dim, descending=descending)


@to_ivy_arrays_and_back
def greater_equal(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.greater_equal(input, other, out=out)


ge = greater_equal


@to_ivy_arrays_and_back
def greater(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.greater(input, other, out=out)


gt = greater


@to_ivy_arrays_and_back
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
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


@to_ivy_arrays_and_back
def isfinite(input):
    return ivy.isfinite(input)


@to_ivy_arrays_and_back
def isinf(input):
    return ivy.isinf(input)


@to_ivy_arrays_and_back
def isposinf(input, *, out=None):
    is_inf = ivy.isinf(input)
    pos_sign_bit = ivy.bitwise_invert(ivy.less(input, 0))
    return ivy.logical_and(is_inf, pos_sign_bit, out=out)


@to_ivy_arrays_and_back
def isneginf(input, *, out=None):
    is_inf = ivy.isinf(input)
    neg_sign_bit = ivy.less(input, 0)
    return ivy.logical_and(is_inf, neg_sign_bit, out=out)


@to_ivy_arrays_and_back
# TODO: the original torch.sort places * right before `out`
def sort(input, *, dim=-1, descending=False, stable=False, out=None):
    values = ivy.sort(input, axis=dim, descending=descending, stable=stable, out=out)
    indices = ivy.argsort(input, axis=dim, descending=descending)
    return namedtuple("sort", ["values", "indices"])(values, indices)


@to_ivy_arrays_and_back
def isnan(input):
    return ivy.isnan(input)


@to_ivy_arrays_and_back
def less_equal(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.less_equal(input, other, out=out)


le = less_equal


@to_ivy_arrays_and_back
def less(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.less(input, other, out=out)


lt = less


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def not_equal(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.not_equal(input, other, out=out)


ne = not_equal


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    input_elements_copy = ivy.reshape(ivy.to_ivy(elements), (-1,))
    test_elements_copy = ivy.reshape(ivy.to_ivy(test_elements), (-1,))

    if (
        ivy.shape(test_elements_copy)[0]
        < 10 * ivy.shape(input_elements_copy)[0] ** 0.145
    ):
        if invert:
            mask = ivy.ones(ivy.shape(input_elements_copy)[0], dtype=bool)
            for a in test_elements_copy:
                mask &= input_elements_copy != a
        else:
            mask = ivy.zeros(ivy.shape(input_elements_copy)[0], dtype=bool)
            for a in test_elements_copy:
                mask |= input_elements_copy == a
        return ivy.reshape(mask, ivy.shape(elements))

    if not assume_unique:
        input_elements_copy, rev_idx = ivy.unique_inverse(input_elements_copy)
        test_elements_copy = ivy.sort(ivy.unique_values(test_elements_copy))

    ar = ivy.concat((input_elements_copy, test_elements_copy))

    order = ivy.argsort(ar, stable=True)
    sar = ar[order]
    if invert:
        bool_ar = sar[1:] != sar[:-1]
    else:
        bool_ar = sar[1:] == sar[:-1]
    flag = ivy.concat((bool_ar, [invert]))
    ret = ivy.empty(ivy.shape(ar), dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ivy.reshape(
            ret[: ivy.shape(input_elements_copy)[0]], ivy.shape(elements)
        )
    else:
        return ivy.reshape(ret[rev_idx], ivy.shape(elements))


@to_ivy_arrays_and_back
def minimum(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.minimum(input, other, out=out)


@to_ivy_arrays_and_back
def fmax(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.where(
        ivy.bitwise_or(ivy.greater(input, other), ivy.isnan(other)),
        input,
        other,
        out=out,
    )


@to_ivy_arrays_and_back
def fmin(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.where(
        ivy.bitwise_or(ivy.less(input, other), ivy.isnan(other)),
        input,
        other,
        out=out,
    )


@to_ivy_arrays_and_back
def msort(input, *, out=None):
    return ivy.sort(input, axis=0, out=out)


@to_ivy_arrays_and_back
def maximum(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.maximum(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def kthvalue(input, k, dim=-1, keepdim=False, *, out=None):
    sorted_input = ivy.sort(input, axis=dim)
    sort_indices = ivy.argsort(input, axis=dim)

    values = ivy.asarray(
        ivy.gather(sorted_input, ivy.array(k - 1), axis=dim), dtype=input.dtype
    )
    indices = ivy.asarray(
        ivy.gather(sort_indices, ivy.array(k - 1), axis=dim), dtype="int64"
    )

    if keepdim:
        values = ivy.expand_dims(values, axis=dim)
        indices = ivy.expand_dims(indices, axis=dim)

    ret = namedtuple("sort", ["values", "indices"])(values, indices)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
@to_ivy_arrays_and_back
def topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
    if dim is None:
        dim = -1
    return ivy.top_k(input, k, axis=dim, largest=largest, sorted=sorted, out=out)
