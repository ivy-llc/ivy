import ivy


def flip(input, dims):
    return ivy.flip(input, dims)


flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def fliplr(input):
    assert len(input.shape) >= 2, "Requires the tensor to be at least 2-D"
    return ivy.flip(input, (-1,))


fliplr.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
