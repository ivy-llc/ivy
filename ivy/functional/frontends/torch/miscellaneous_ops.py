import ivy


def flipud(input):
    return ivy.flipud(input)


flipud.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)
