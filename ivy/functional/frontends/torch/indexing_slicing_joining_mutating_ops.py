# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)


def swapaxes(input, axis0, axis1):
    return ivy.swapaxes(input, axis0, axis1)


swapaxes.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
