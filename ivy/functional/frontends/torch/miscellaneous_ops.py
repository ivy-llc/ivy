import ivy


def flip(input, dims):
    return ivy.flip(input, axis=dims)


flip.unsupported_dtypes = (
    "uint16",
    "uint32",
    "uint64",
)


def fliplr(input):
    assert len(input.shape) >= 2, "Requires the tensor to be at least 2-D"
    return ivy.flip(input, axis=(-1,))


def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, axis=dims)


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


def diagonal(input, offset=0, dim1=0, dim2=1):
    return ivy.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


def diagflat(input, offset=0):
    input_flat = input.flatten()

    # data = []
    # for i in range(offset):
    #    data += [[0 for j in range(len(input_flat) + abs(offset))]]

    data = [
        [
            input_flat[i] if i < len(input_flat) and i - r == offset else 0
            for i in range(len(input_flat) + abs(offset))
        ]
        for r in range(len(input_flat) + abs(offset))
    ]

    return ivy.array(data, dtype=input.dtype)


diagflat.unsupported_dtypes = ("float16",)
