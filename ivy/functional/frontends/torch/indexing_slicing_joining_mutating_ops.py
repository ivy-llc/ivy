# local
import ivy


def cat(tensors, dim=0, *, out=None):
    return ivy.concat(tensors, dim, out=out)


def permute(input, dims):
    return ivy.permute_dims(input, dims)


permute.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def movedim(input, source, destination):
    return ivy.movedim(input, source, destination)


movedim.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
