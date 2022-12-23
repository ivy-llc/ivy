# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def all(
    a,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=None,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def any(
    a,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=None,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    ret = ivy.any(a, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@to_ivy_arrays_and_back
def isscalar(element):
    return (
        isinstance(element, int)
        or isinstance(element, bool)
        or isinstance(element, float)
    )
