# global
import warnings

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    outputs_to_frontend_arrays,
)
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_dtype


@to_ivy_arrays_and_back
def all(a, axis=None, out=None, keepdims=False, *, where=False):
    return ivy.all(a, axis=axis, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def argmax(a, axis=None, out=None, keepdims=False):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def argwhere(a, /, *, size=None, fill_value=None):
    if size is None and fill_value is None:
        return ivy.argwhere(a)

    result = ivy.matrix_transpose(
        ivy.vstack(ivy.nonzero(a, size=size, fill_value=fill_value))
    )
    num_of_dimensions = a.ndim

    if num_of_dimensions == 0:
        return result[:0].reshape(result.shape[0], 0)

    return result.reshape(result.shape[0], num_of_dimensions)


@to_ivy_arrays_and_back
def argsort(a, axis=-1, kind="stable", order=None):
    if kind != "stable":
        warnings.warn(
            "'kind' argument to argsort is ignored; only 'stable' sorts "
            "are supported."
        )
    if order is not None:
        raise ivy.exceptions.IvyError("'order' argument to argsort is not supported.")

    return ivy.argsort(a, axis=axis)


@to_ivy_arrays_and_back
def fmax(x1, x2):
    ret = ivy.where(
        ivy.bitwise_or(ivy.greater(x1, x2), ivy.isnan(x2)),
        x1,
        x2,
    )
    return ret


@to_ivy_arrays_and_back
def bitwise_and(x1, x2):
    return ivy.bitwise_and(x1, x2)


@to_ivy_arrays_and_back
def bitwise_not(x):
    return ivy.bitwise_invert(x)


@to_ivy_arrays_and_back
def bitwise_or(x1, x2):
    return ivy.bitwise_or(x1, x2)


@to_ivy_arrays_and_back
def bitwise_xor(x1, x2):
    return ivy.bitwise_xor(x1, x2)


@to_ivy_arrays_and_back
def heaviside(x1, x2):
    return ivy.heaviside(x1, x2)


@to_ivy_arrays_and_back
def any(a, axis=None, out=None, keepdims=False, *, where=None):
    # TODO: Out not supported
    ret = ivy.any(a, axis=axis, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(None, ivy.zeros_like(ret)))
    return ret


@handle_numpy_dtype
@to_ivy_arrays_and_back
def hstack(tup, dtype=None):
    # TODO: dtype supported in JAX v0.3.20
    return ivy.hstack(tup)


@to_ivy_arrays_and_back
def maximum(x1, x2):
    return ivy.maximum(x1, x2)


@to_ivy_arrays_and_back
def minimum(x1, x2):
    return ivy.minimum(x1, x2)


alltrue = all


sometrue = any


@to_ivy_arrays_and_back
def diagonal(a, offset=0, axis1=0, axis2=1):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def eye(N, M=None, k=0, dtype=None):
    return ivy.eye(N, M, k=k, dtype=dtype)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def zeros_like(a, dtype=None, shape=None):
    if shape:
        return ivy.zeros(shape, dtype=dtype)
    return ivy.zeros_like(a, dtype=dtype)


@to_ivy_arrays_and_back
def msort(a):
    return ivy.msort(a)


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
