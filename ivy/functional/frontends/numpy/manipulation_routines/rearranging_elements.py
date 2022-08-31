# local
import ivy


def flip(m, axis=None):
    return ivy.flip(m, axis=axis, out=None)
