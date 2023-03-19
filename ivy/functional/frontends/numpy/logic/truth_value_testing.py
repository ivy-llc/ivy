# global
import ivy
import numpy as np
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


@handle_numpy_out
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


@handle_numpy_out
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
        or isinstance(element, complex)
    )


@to_ivy_arrays_and_back
def isfortran(a: np.ndarray):
    return a.flags.fnc


@to_ivy_arrays_and_back
def isreal(x):
    return ivy.isreal(x)


@to_ivy_arrays_and_back
def isrealobj(x: any):
    return not ivy.is_complex_dtype(ivy.dtype(x))


@to_ivy_arrays_and_back
def iscomplexobj(a: np.ndarray):
    """The return value, True if x is of a complex type or 
        has at least one complex element.
    Args:
        a (np.ndarray): _description_
    """
    for ele in a:
        # ivy.dtype considers a+0j also as complex,
        # which is same requirement as of iscomplexobj()
        if ivy.is_complex_dtype(ivy.dtype(ele)):
            return True
        else:
            return False


@to_ivy_arrays_and_back
def iscomplex(x: np.ndarray):
    return ivy.bitwise_invert(ivy.isreal(x))
