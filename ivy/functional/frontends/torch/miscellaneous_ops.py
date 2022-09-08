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

    output_size = len(input_flat) + abs(offset)

    ret = ivy.zeros((output_size, output_size), dtype=input.dtype)

    index = 0
    for i in range(0, len(input_flat)):
        if offset >= 0:
            ret[i][i + offset] = input_flat[index]
        else:
            ret[i - offset][i] = input_flat[index]
        index += 1

    return ret


diagflat.unsupported_dtypes = ("float16",)


def diag(input, diagonal=0, *, out=None):
    if len(input.shape) == 1:
        ret = diagflat(input, offset=diagonal)
    if len(input.shape) == 2:
        ret = ivy.diagonal(input, offset=diagonal)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)

    return ret


diag.unsupported_dtypes = ("float16",)
