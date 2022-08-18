import ivy


def flip(input, dims):
    return ivy.flip(input, dims)


flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, dims)


roll.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
