import ivy


def flip(input, dims):
    return ivy.flip(input, dims)


flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
