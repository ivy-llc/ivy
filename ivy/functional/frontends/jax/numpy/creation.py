import ivy
from ivy.functional.frontends.jax.devicearray import DeviceArray

from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_frontend_arrays,
    handle_jax_dtype,
)


@handle_jax_dtype
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
    if ret.shape == () and dtype is None:
        return DeviceArray(ret, weak_type=True)
    return DeviceArray(ret)


@handle_jax_dtype
@to_ivy_arrays_and_back
def zeros_like(a, dtype=None, shape=None):
    if shape:
        return ivy.zeros(shape, dtype=dtype)
    return ivy.zeros_like(a, dtype=dtype)


@handle_jax_dtype
@outputs_to_frontend_arrays
def arange(start, stop=None, step=1, dtype=None):
    return ivy.arange(start, stop, step=step, dtype=dtype)


@handle_jax_dtype
@to_ivy_arrays_and_back
def zeros(shape, dtype=None):
    return DeviceArray(ivy.zeros(shape, dtype=dtype))


@handle_jax_dtype
@to_ivy_arrays_and_back
def ones(shape, dtype=None):
    return DeviceArray(ivy.ones(shape, dtype=dtype))


@handle_jax_dtype
@to_ivy_arrays_and_back
def ones_like(a, dtype=None, shape=None):
    if shape:
        return ivy.ones(shape, dtype=dtype)
    return ivy.ones_like(a, dtype=dtype)


@handle_jax_dtype
@to_ivy_arrays_and_back
def asarray(a, dtype=None, order=None):
    return array(a, dtype=dtype, order=order)


@handle_jax_dtype
@to_ivy_arrays_and_back
def hstack(tup, dtype=None):
    # TODO: dtype supported in JAX v0.3.20
    return ivy.hstack(tup)


@handle_jax_dtype
@to_ivy_arrays_and_back
def eye(N, M=None, k=0, dtype=None):
    return DeviceArray(ivy.eye(N, M, k=k, dtype=dtype))


@to_ivy_arrays_and_back
def triu(m, k=0):
    return ivy.triu(m, k=k)


@handle_jax_dtype
@to_ivy_arrays_and_back
def empty(shape, dtype=None):
    return DeviceArray(ivy.empty(shape, dtype=dtype))


@to_ivy_arrays_and_back
def vander(x, N=None, increasing=False):
    if N == 0:
        return ivy.array([], dtype=x.dtype)
    else:
        return ivy.vander(x, N=N, increasing=increasing, out=None)


@to_ivy_arrays_and_back
def full_like(a, fill_value, dtype=None, shape=None):
    return ivy.full_like(a, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def ndim(a):
    return ivy.astype(ivy.array(a.ndim), ivy.int64)
