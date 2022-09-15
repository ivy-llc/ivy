import ivy


def flip(input, dims):
    return ivy.flip(input, axis=dims)


def fliplr(input):
    assert len(input.shape) >= 2, "Requires the tensor to be at least 2-D"
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


def triu(input, diagonal=0, *, out=None):
    return ivy.triu(input, k=diagonal, out=out)


# THIS SHOULD NOT BE COMMITTED AS PART OF THE FRONTEND_TEST_TYPE_DETECTION BRANCH
# ITS JUST HERE TO TEST THE FIXES IN THAT BRANCH
# IT SHOULD BE COMMITTED FROM THE TRACE BRANCH
def trace(input):
    if "int" in input.dtype:
        input = input.astype("int64")
    target_type = "int64" if "int" in input.dtype else input.dtype
    return ivy.astype(ivy.trace(input), target_type)


trace.unsupported_dtypes = ("float16",)
