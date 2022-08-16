# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)


def concat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)


def permute(input, dims):
    return ivy.permute_dims(input, dims)


permute.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def swapdims(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


swapdims.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def reshape(input, shape):
    return ivy.reshape(input, shape)


reshape.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
