# local
import ivy


def reshape(input, shape):
    return ivy.reshape(input, shape)

reshape.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
