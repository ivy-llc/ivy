import ivy


def flip(input, dims):
    return ivy.flip(input, dims)

flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def fliplr(input):
    return ivy.fliplr(input)

fliplr.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def flipud(input):
    return ivy.flipud(input)

flipud.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
