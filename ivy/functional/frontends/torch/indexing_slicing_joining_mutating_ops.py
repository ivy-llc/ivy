# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)


def transpose(input, dim0, dim1):
    return ivy.swapaxes(input, dim0, dim1)


transpose.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
