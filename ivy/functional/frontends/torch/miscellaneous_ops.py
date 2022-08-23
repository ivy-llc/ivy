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

def cumsum(input, dim, *, dtype=None, out=None):
    return ivy.cumsum(x=input, axis=dim, out=out)

cumsum.unsupported_dtypes = (
    "float16",
    "uint16",
    "uint32",
    "uint64",
)