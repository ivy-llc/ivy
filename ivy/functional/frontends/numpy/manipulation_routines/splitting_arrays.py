import ivy


def split(a, splits, axis=0):
    return ivy.split(a, /, *, splits=None, axis=0, with_remainder=False)
