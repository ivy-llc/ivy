# local
import ivy


def empty(shape, dtype="float64", order="C", *, like=None):
    return ivy.empty(shape, dtype=dtype)


def empty_like(prototype, dtype=None, order="K", subok=True, shape=None):
    if shape:
        return ivy.empty(shape, dtype=dtype)
    return ivy.empty_like(prototype, dtype=dtype)


def eye(N, M=None, k=0, dtype="float64", order="C", *, like=None):
    return ivy.eye(N, M, k=k, dtype=dtype)


def identity(n, dtype=None, *, like=None):
    return ivy.eye(n, dtype=dtype)


def full(shape, fill_value, dtype=None, order="C", *, like=None):
    return ivy.full(shape, fill_value, dtype=dtype)
