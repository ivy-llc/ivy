# local
from ivy.functional.frontends.numpy.creation_routines import asarray as as_array


def asarray(a, dtype=None, order=None, *, like=None):
    return as_array(a, dtype, order, like=like)
