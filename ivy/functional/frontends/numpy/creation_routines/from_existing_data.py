import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def array(object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    ret = ivy.array(object, copy=copy, dtype=dtype)
    if ivy.get_num_dims(ret) < ndmin:
        ret = ivy.expand_dims(ret, axis=list(range(ndmin - ivy.get_num_dims(ret))))
    return ret


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


@handle_numpy_dtype
def frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None):
    return ivy.frombuffer(buffer)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def fromstring(string, dtype=float, count=-1, *, sep, like=None):
    try:
        split_string = string.split(sep)
    except ValueError:
        split_string = string

    string_to_array = ivy.array(split_string, dtype=dtype)
    bytes_string = string_to_array.tobytes("C")

    return ivy.frombuffer(bytes_string, dtype=dtype, count=count)
