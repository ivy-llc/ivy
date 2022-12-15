# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_dtype


@to_ivy_arrays_and_back
def fmax(x1, x2):
    ret = ivy.where(
        ivy.bitwise_or(ivy.greater(x1, x2), ivy.isnan(x2)),
        x1,
        x2,
    )
    return ret


@to_ivy_arrays_and_back
def heaviside(x1, x2):
    return ivy.heaviside(x1, x2)


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
