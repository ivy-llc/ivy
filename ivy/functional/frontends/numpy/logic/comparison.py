# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    inputs_to_ivy_arrays,
    handle_numpy_casting,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def equal(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.equal(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@to_ivy_arrays_and_back
def array_equal(a1, a2, equal_nan=False):
    if not equal_nan:
        return ivy.array(ivy.array_equal(a1, a2))
    a1nan, a2nan = ivy.isnan(a1), ivy.isnan(a2)
    if not (a1nan == a2nan).all():
        return False
    return ivy.array(ivy.array_equal(a1[~a1nan], a2[~a2nan]))


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def greater(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.greater(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def greater_equal(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.greater_equal(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def less(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.less(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def less_equal(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.less_equal(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def not_equal(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = ivy.not_equal(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@inputs_to_ivy_arrays
def array_equiv(a1, a2):
    if len(ivy.shape(a1)) < len(ivy.shape(a2)):
        a1 = ivy.broadcast_to(a1, ivy.shape(a2))
    else:
        a2 = ivy.broadcast_to(a2, ivy.shape(a1))
    return ivy.array(ivy.array_equal(a1, a2))
