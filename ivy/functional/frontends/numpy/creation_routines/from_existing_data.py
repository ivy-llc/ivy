import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def array(object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    if dtype:
        return ivy.array(object, copy=copy, dtype=dtype)
    return ivy.array(object, copy=copy, dtype=object.dtype())


@to_ivy_arrays_and_back
def asarray(
    a,
    dtype=None,
    order=None,
    *,
    like=None,
):
    if dtype:
        return ivy.asarray(a, dtype=dtype)
    return ivy.asarray(a, dtype=a.dtype())


@to_ivy_arrays_and_back
def copy(a, order="K", subok=False):
    return ivy.copy_array(a)
