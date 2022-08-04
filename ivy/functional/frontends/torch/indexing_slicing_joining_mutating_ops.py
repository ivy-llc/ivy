# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)


def concat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)


def stack(tensors, dim=0, *, out=None):
    return ivy.stack(tensors, dim, out=out)


def reshape(input, shape):
    return ivy.expand_dims(input, shape)

reshape.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
