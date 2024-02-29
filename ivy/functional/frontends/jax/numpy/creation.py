import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.jax.array import Array
import ivy.functional.frontends.jax.numpy as jnp_frontend
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_frontend_arrays,
    handle_jax_dtype,
    inputs_to_ivy_arrays,
)

from ivy.func_wrapper import handle_out_argument
from ivy import with_unsupported_device_and_dtypes

ndarray = Array


@with_unsupported_device_and_dtypes(
    {
        "0.4.24 and below": {
            "cpu": (
                "float16",
                "bflooat16",
                "complex64",
                "complex128",
            ),
            "gpu": (
                "complex64",
                "complex128",
            ),
        }
    },
    "jax",
)
@handle_jax_dtype
@outputs_to_frontend_arrays
def arange(start, stop=None, step=1, dtype=None):
    return ivy.arange(start, stop, step=step, dtype=dtype)


@handle_jax_dtype
@to_ivy_arrays_and_back
def array(object, dtype=None, copy=True, order="K", ndmin=0):
    if order is not None and order != "K":
        raise ivy.utils.exceptions.IvyNotImplementedException(
            "Only implemented for order='K'"
        )
    device = ivy.default_device()
    if ivy.is_array(object):
        device = ivy.dev(object)
    ret = ivy.array(object, dtype=dtype, device=device)
    if ivy.get_num_dims(ret) < ndmin:
        ret = ivy.expand_dims(ret, axis=list(range(ndmin - ivy.get_num_dims(ret))))

    if ret.shape == () and dtype is None:
        return Array(ret, weak_type=True)
    return Array(ret)


@handle_jax_dtype
@to_ivy_arrays_and_back
def asarray(a, dtype=None, order=None):
    return array(a, dtype=dtype, order=order)


@to_ivy_arrays_and_back
def bool_(x):
    return ivy.astype(x, ivy.bool)


@to_ivy_arrays_and_back
def cdouble(x):
    return ivy.astype(x, ivy.complex128)


@to_ivy_arrays_and_back
@handle_out_argument
def compress(condition, a, *, axis=None, out=None):
    condition_arr = ivy.asarray(condition).astype(bool)
    if condition_arr.ndim != 1:
        raise ivy.utils.exceptions.IvyException("Condition must be a 1D array")
    if axis is None:
        arr = ivy.asarray(a).flatten()
        axis = 0
    else:
        arr = ivy.moveaxis(a, axis, 0)
    if condition_arr.shape[0] > arr.shape[0]:
        raise ivy.utils.exceptions.IvyException(
            "Condition contains entries that are out of bounds"
        )
    arr = arr[: condition_arr.shape[0]]
    return ivy.moveaxis(arr[condition_arr], 0, axis)


@to_ivy_arrays_and_back
def copy(a, order=None):
    return array(a, order=order)


@to_ivy_arrays_and_back
def csingle(x):
    return ivy.astype(x, ivy.complex64)


@to_ivy_arrays_and_back
def double(x):
    return ivy.astype(x, ivy.float64)


@handle_jax_dtype
@to_ivy_arrays_and_back
def empty(shape, dtype=None):
    return Array(ivy.empty(shape=shape, dtype=dtype))


@handle_jax_dtype
@to_ivy_arrays_and_back
def empty_like(prototype, dtype=None, shape=None):
    # XLA cannot create uninitialized arrays
    # jax.numpy.empty_like returns an array initialized with zeros.
    if shape:
        return ivy.zeros(shape, dtype=dtype)
    return ivy.zeros_like(prototype, dtype=dtype)


@handle_jax_dtype
@to_ivy_arrays_and_back
def eye(N, M=None, k=0, dtype=None):
    return Array(ivy.eye(N, M, k=k, dtype=dtype))


@to_ivy_arrays_and_back
def from_dlpack(x):
    return ivy.from_dlpack(x)


@to_ivy_arrays_and_back
def frombuffer(buffer, dtype="float", count=-1, offset=0):
    return ivy.frombuffer(buffer, dtype, count, offset)


@to_ivy_arrays_and_back
def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def full_like(a, fill_value, dtype=None, shape=None):
    return ivy.full_like(a, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    cr = ivy.log(stop / start) / (num - 1 if endpoint else num)
    x = ivy.linspace(
        0, cr * (num - 1 if endpoint else num), num, endpoint=endpoint, axis=axis
    )
    x = ivy.exp(x)
    x = start * x
    x[0] = (start * cr) / cr
    if endpoint:
        x[-1] = stop
    return x.asarray(dtype=dtype)


@handle_jax_dtype
@to_ivy_arrays_and_back
def hstack(tup, dtype=None):
    # TODO: dtype supported in JAX v0.3.20
    return ivy.hstack(tup)


@handle_jax_dtype
@to_ivy_arrays_and_back
def identity(n, dtype=None):
    return ivy.eye(n, dtype=dtype)


@to_ivy_arrays_and_back
def in1d(ar1, ar2, assume_unique=False, invert=False):
    del assume_unique
    ar1_flat = ivy.flatten(ar1)
    ar2_flat = ivy.flatten(ar2)
    if invert:
        return (ar1_flat[:, None] != ar2_flat[None, :]).all(axis=-1)
    else:
        return (ar1_flat[:, None] == ar2_flat[None, :]).any(axis=-1)


@inputs_to_ivy_arrays
def iterable(y):
    return hasattr(y, "__iter__") and y.ndim > 0


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    ret = ivy.linspace(start, stop, num, axis=axis, endpoint=endpoint, dtype=dtype)
    if retstep:
        if endpoint:
            num -= 1
        step = ivy.divide(ivy.subtract(stop, start), num)
        return ret, step
    return ret


@handle_jax_dtype
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if not endpoint:
        interval = (stop - start) / num
        stop -= interval
    return ivy.logspace(start, stop, num, base=base, axis=axis, dtype=dtype)


@to_ivy_arrays_and_back
def meshgrid(*x, copy=True, sparse=False, indexing="xy"):
    # TODO: handle 'copy' argument when ivy.meshgrid supports it
    ivy_meshgrid = ivy.meshgrid(*x, sparse=sparse, indexing=indexing)
    return ivy_meshgrid


@to_ivy_arrays_and_back
def ndim(a):
    if not isinstance(a, ivy.Array):
        return 0
    return ivy.astype(ivy.array(a.ndim), ivy.int64)


@handle_jax_dtype
@to_ivy_arrays_and_back
def ones(shape, dtype=None):
    return Array(ivy.ones(shape, dtype=dtype))


@handle_jax_dtype
@to_ivy_arrays_and_back
def ones_like(a, dtype=None, shape=None):
    if shape:
        return ivy.ones(shape, dtype=dtype)
    return ivy.ones_like(a, dtype=dtype)


@to_ivy_arrays_and_back
def setdiff1d(ar1, ar2, assume_unique=False, *, size=None, fill_value=None):
    fill_value = ivy.array(0 if fill_value is None else fill_value, dtype=ar1.dtype)
    if ar1.size == 0:
        return ivy.full(size or 0, fill_value, dtype=ar1.dtype)
    if not assume_unique:
        val = (
            ivy.to_scalar(ivy.all(ar1))
            if ivy.is_bool_dtype(ar1.dtype)
            else ivy.to_scalar(ivy.min(ar1))
        )
        ar1 = jnp_frontend.unique(ar1, size=size and ar1.size, fill_value=val).ivy_array
    mask = in1d(ar1, ar2, invert=True).ivy_array
    if size is None:
        return ar1[mask]
    else:
        if not assume_unique:
            # Set mask to zero at locations corresponding to unique() padding.
            n_unique = ar1.size + 1 - (ar1 == ar1[0]).sum(dtype=ivy.int64)
            mask = ivy.where(ivy.arange(ar1.size) < n_unique, mask, False)
        return ivy.where(
            ivy.arange(size) < mask.sum(dtype=ivy.int64),
            ar1[jnp_frontend.where(mask, size=size)[0].ivy_array],
            fill_value,
        )


@to_ivy_arrays_and_back
def single(x):
    return ivy.astype(x, ivy.float32)


@to_ivy_arrays_and_back
def size(a, axis=None):
    ivy.set_default_int_dtype("int64")
    if axis is not None:
        sh = ivy.shape(a)
        return sh[axis]
    return a.size


@to_ivy_arrays_and_back
def triu(m, k=0):
    return ivy.triu(m, k=k)


@to_ivy_arrays_and_back
def vander(x, N=None, increasing=False):
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")
    if N == 0:
        return ivy.array([], dtype=x.dtype).reshape((x.shape[0], 0))
    else:
        return ivy.vander(x, N=N, increasing=increasing)


@handle_jax_dtype
@to_ivy_arrays_and_back
def zeros(shape, dtype=None):
    return Array(ivy.zeros(shape, dtype=dtype))


@handle_jax_dtype
@to_ivy_arrays_and_back
def zeros_like(a, dtype=None, shape=None):
    if shape:
        return ivy.zeros(shape, dtype=dtype)
    return ivy.zeros_like(a, dtype=dtype)
