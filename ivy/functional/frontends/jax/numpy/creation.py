import ivy
from ivy.functional.frontends.jax.devicearray import DeviceArray
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_dtype
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_frontend_arrays,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def array(object, dtype=None, copy=True, order="K", ndmin=0):
    # TODO must ensure the array is created on default device.
    if order is not None and order != "K":
        raise ivy.exceptions.IvyNotImplementedException(
            "Only implemented for order='K'"
        )
    ret = ivy.array(object, dtype=dtype)
    if ivy.get_num_dims(ret) < ndmin:
        ret = ivy.expand_dims(ret, axis=list(range(ndmin - ivy.get_num_dims(ret))))
    return DeviceArray(ret)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def zeros_like(a, dtype=None, shape=None):
    if shape:
        return ivy.zeros(shape, dtype=dtype)
    return ivy.zeros_like(a, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def arange(start, stop=None, step=1, dtype=None):
    return ivy.arange(start, stop, step=step, dtype=dtype)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def zeros(shape, dtype=None):
    if dtype is None:
        dtype = ivy.float64
    return ivy.zeros(shape, dtype=dtype)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def ones(shape, dtype=None):
    return ivy.ones(shape, dtype=dtype)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def ones_like(a, dtype=None, shape=None):
    if shape:
        return ivy.ones(shape, dtype=dtype)
    return ivy.ones_like(a, dtype=dtype)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def asarray(a, dtype=None, order=None):
    return ivy.asarray(a, dtype=dtype)


@to_ivy_arrays_and_back
def uint16(x):
    return ivy.astype(x, ivy.UintDtype("uint16"), copy=False)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def hstack(tup, dtype=None):
    # TODO: dtype supported in JAX v0.3.20
    return ivy.hstack(tup)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def eye(N, M=None, k=0, dtype=None):
    return ivy.eye(N, M, k=k, dtype=dtype)


@to_ivy_arrays_and_back
def triu(m, k=0):
    return ivy.triu(m, k=k)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def empty(shape, dtype=None):
    return ivy.empty(shape, dtype=dtype)
