import ivy


def argmax(input, dim=None, keepdim=False):
    return ivy.argmax(input, axis=dim, keepdims=keepdim)


def argmin(input, dim=None, keepdim=False):
    return ivy.argmin(input, axis=dim, keepdims=keepdim)


def amax(input, dim, keepdim=False, *, out=None):
    return ivy.max(input, axis=dim, keepdims=keepdim, out=out)


def amin(input, dim, keepdim=False, *, out=None):
    return ivy.min(input, axis=dim, keepdims=keepdim, out=out)
