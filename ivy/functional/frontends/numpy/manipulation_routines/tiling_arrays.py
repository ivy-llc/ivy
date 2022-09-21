# local
import ivy


def tile(A, reps):
    return ivy.tile(A, reps)


def repeat(a, repeats, axis=None):
    return ivy.repeat(a, repeats, axis=axis)
