# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    outputs_to_frontend_arrays,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def empty(shape, dtype="float64", order="C", *, like=None):
    return ivy.empty(shape=shape, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def empty_like(prototype, dtype=None, order="K", subok=True, shape=None):
    if shape:
        return ivy.empty(shape=shape, dtype=dtype)
    return ivy.empty_like(prototype, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def eye(N, M=None, k=0, dtype="float64", order="C", *, like=None):
    return ivy.eye(N, M, k=k, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def identity(n, dtype=None, *, like=None):
    return ivy.eye(n, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def ones(shape, dtype=None, order="C", *, like=None):
    return ivy.ones(shape, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def ones_like(a, dtype=None, order="K", subok=True, shape=None):
    if shape:
        return ivy.ones(shape, dtype=dtype)
    return ivy.ones_like(a, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def zeros(shape, dtype=float, order="C", *, like=None):
    return ivy.zeros(shape, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def zeros_like(a, dtype=None, order="K", subok=True, shape=None):
    if shape:
        return ivy.zeros(shape, dtype=dtype)
    return ivy.zeros_like(a, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def full(shape, fill_value, dtype=None, order="C", *, like=None):
    return ivy.full(shape, fill_value, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    if shape:
        return ivy.full(shape, fill_value, dtype=dtype)
    return ivy.full_like(a, fill_value, dtype=dtype)


@handle_numpy_dtype
@outputs_to_frontend_arrays
def fromfunction(function, shape, *, dtype="float64", like=None):
    base = ivy.empty(shape=shape, dtype=dtype)
    for pos, _ in ivy.ndenumerate(base):
        base[pos] = function(*pos)
    return base
