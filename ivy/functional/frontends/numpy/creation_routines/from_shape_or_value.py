# local
import ivy


def full(shape, fill_value, dtype=None, order="C", *, like=None):
    return ivy.full(shape, fill_value, dtype=dtype)
