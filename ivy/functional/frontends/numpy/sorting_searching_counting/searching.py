# local

import ivy

from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs

from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


# --- Helpers --- #
# --------------- #


# nanargmin and nanargmax composition helper
def _nanargminmax(a, axis=None):
    # check nans
    nans = ivy.isnan(a).astype(ivy.bool)
    # replace nans with inf
    a = ivy.where(nans, ivy.inf, a)
    if nans is not None:
        nans = ivy.all(nans, axis=axis)
        if ivy.any(nans):
            raise ivy.utils.exceptions.IvyError("All-NaN slice encountered")
    return a


# --- Main --- #
# ------------ #


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def argmax(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
):
    return ivy.argmax(a, axis=axis, out=out, keepdims=keepdims)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def argmin(a, /, *, axis=None, keepdims=False, out=None):
    return ivy.argmin(a, axis=axis, out=out, keepdims=keepdims)


@to_ivy_arrays_and_back
def argwhere(a):
    return ivy.argwhere(a)


@to_ivy_arrays_and_back
def extract(cond, arr, /):
    if cond.dtype == "bool":
        return arr[cond]
    else:
        return arr[cond != 0]


@to_ivy_arrays_and_back
def flatnonzero(a):
    return ivy.nonzero(ivy.reshape(a, (-1,)))


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmax(a, /, *, axis=None, out=None, keepdims=False):
    a = _nanargminmax(a, axis=axis)
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmin(a, /, *, axis=None, out=None, keepdims=False):
    a = _nanargminmax(a, axis=axis)
    return ivy.argmin(a, axis=axis, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def nonzero(a):
    return ivy.nonzero(a)


@to_ivy_arrays_and_back
def searchsorted(a, v, side="left", sorter=None):
    return ivy.searchsorted(a, v, side=side, sorter=sorter)


@to_ivy_arrays_and_back
def where(cond, x1=None, x2=None, /):
    if x1 is None and x2 is None:
        # numpy where behaves as np.asarray(condition).nonzero() when x and y
        # not included
        return ivy.asarray(cond).nonzero()
    elif x1 is not None and x2 is not None:
        x1, x2 = promote_types_of_numpy_inputs(x1, x2)
        return ivy.where(cond, x1, x2)
    else:
        raise ivy.utils.exceptions.IvyException("where takes either 1 or 3 arguments")
