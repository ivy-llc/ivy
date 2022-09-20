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


def tril(input, diagonal=0, *, out=None):
    return ivy.tril(input, k=diagonal, out=out)


def flatten(input, start_dim=0, end_dim=-1):

    # This loop is to work out the new shape
    # It is a map f: Z^n -> Z^(n-1) where
    # (...., a,b, ....)
    # maps to
    # (...., ab, ....)
    # iteratively, then resize the array.

    new_shape = list(input.shape)

    if end_dim == -1:
        end_dim = len(new_shape) - 1

    for i in range(start_dim, end_dim):
        new_shape[start_dim] = new_shape[start_dim] * new_shape[start_dim + 1]
        for j in range(start_dim + 1, len(new_shape) - 1, 1):
            new_shape[j] = new_shape[j + 1]
        new_shape = new_shape[:-1]

    input = ivy.reshape(input, shape=new_shape)
    return input
    