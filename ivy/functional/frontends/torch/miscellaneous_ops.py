import ivy


def flip(input, dims):
    return ivy.flip(input, dims)


flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def flipud(input):
    assert len(input.shape) >= 1, "Requires the tensor to be at least 1-D"
    return ivy.flip(input, (0,))


flipud.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)