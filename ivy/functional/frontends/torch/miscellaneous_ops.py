import ivy


def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, dims)


roll.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
