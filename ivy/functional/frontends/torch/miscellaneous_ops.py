import ivy


def flip(input, dims):
    return ivy.flip(input, axis=dims)


def fliplr(input):
    ivy.assertions.check_greater(
        len(input.shape),
        2,
        allow_equal=True,
        message="requires tensor to be at least 2D",
    )
    return ivy.flip(input, axis=(-1,))


def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, axis=dims)


def cumsum(input, dim, *, dtype=None, out=None):
    return ivy.cumsum(input, axis=dim, dtype=dtype, out=out)


def tril_indices(row, col, offset=0, *, dtype="int64", device="cpu", layout=None):
    sample_matrix = ivy.tril(ivy.ones((row, col), device=device), k=offset)
    return ivy.stack(ivy.nonzero(sample_matrix)).astype(dtype)


def cumprod(input, dim, *, dtype=None, out=None):
    return ivy.cumprod(input, axis=dim, dtype=dtype, out=out)


def diagonal(input, offset=0, dim1=0, dim2=1):
    return ivy.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


def triu_indices(row, col, offset=0, dtype="int64", device="cpu", layout=None):
    # TODO: Handle layout flag when possible.
    sample_matrix = ivy.triu(ivy.ones((row, col), device=device), k=offset)
    return ivy.stack(ivy.nonzero(sample_matrix)).astype(dtype)


def triu(input, diagonal=0, *, out=None):
    return ivy.triu(input, k=diagonal, out=out)
