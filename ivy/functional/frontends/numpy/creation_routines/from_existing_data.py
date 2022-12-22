import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def array(object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    return ivy.array(object, copy=copy, dtype=dtype)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def asarray(
    a,
    dtype=None,
    order=None,
    *,
    like=None,
):
    return ivy.asarray(a, dtype=dtype)


@to_ivy_arrays_and_back
def copy(a, order="K", subok=False):
    return ivy.copy_array(a)
