# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    inputs_to_ivy_arrays,
)
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs


@to_ivy_arrays_and_back
def isneginf(x, out=None):
    isinf = ivy.isinf(x)
    neg_sign_bit = ivy.less(x, 0)
    return ivy.logical_and(isinf, neg_sign_bit, out=out)


@to_ivy_arrays_and_back
def isposinf(x, out=None):
    isinf = ivy.isinf(x)
    pos_sign_bit = ivy.greater(x, 0)
    return ivy.logical_and(isinf, pos_sign_bit, out=out)


@inputs_to_ivy_arrays
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@inputs_to_ivy_arrays
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


# TODO: datetime dtypes not supported by ivy yet
def isnat(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.isnat(x, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret
