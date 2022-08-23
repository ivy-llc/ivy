# local
import ivy


def empty(shape, dtype="float64", order="C", *, like=None):
    return ivy.empty(shape, dtype=dtype)


def full(shape, fill_value, dtype=None, order="C", *, like=None):
    return ivy.full(shape, fill_value, dtype=dtype)
