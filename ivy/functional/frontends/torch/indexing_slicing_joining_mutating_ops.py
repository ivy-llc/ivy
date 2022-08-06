# local
import ivy


def reshape(input, shape, copy=None):
    return ivy.reshape(input, shape, copy=copy)


reshape.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
