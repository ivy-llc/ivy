# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, axis=dim, out=out)


def concat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, axis=dim, out=out)


def chunk(input, chunks, dim=0):
    return ivy.split(input, num_or_size_splits=chunks, dim=dim, with_remainder=True)
    

def permute(input, dims):
    return ivy.permute_dims(input, axes=dims)


def swapdims(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


swapdims.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def transpose(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


def reshape(input, shape):
    return ivy.reshape(input, shape)


def stack(tensors, dim=0, *, out=None):
    return ivy.stack(tensors, axis=dim, out=out)