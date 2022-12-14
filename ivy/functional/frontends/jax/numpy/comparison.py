# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return ivy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


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
def isneginf(x, out=None):
    return ivy.isneginf(x, out=out)


@to_ivy_arrays_and_back
def isposinf(x, out=None):
    return ivy.isposinf(x, out=out)


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
