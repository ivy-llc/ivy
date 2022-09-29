# local
import ivy


def roll(a, shift, axis=None):
    return ivy.roll(a, shift, axis=axis)


def flip(m, axis=None):
    return ivy.flip(m, axis=axis, out=None)
