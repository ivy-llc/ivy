import ivy


def flip(input, dims):
    return ivy.flip(input, dims)


def cumsum(input, dim, *, dtype=None, out=None):
    return ivy.cumsum(x=input, axis=dim, out=out)
