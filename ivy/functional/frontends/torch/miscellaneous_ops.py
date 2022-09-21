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


def renorm(input, p, dim, maxnorm, *, out=None):

    # ivy.map
    # return ivy.clip_vector_norm(input, maxnorm, p=p, axis=dim, out=out)

    input_dtype = input.dtype
    input = ivy.astype(input, "float64")
    # Avoids division by 0 error later.
    if maxnorm == 0:
        # if False:
        ret = ivy.zeros(input.shape, dtype=input_dtype)
    else:
        # The p-norm of a single value is that value
        # if len(input.shape) == 1:
        #    norms = input

        # The p-norm of a single value is that value
        # This avoids the case where a 0 dimensional array presents an indexing error
        # E.g. axis 0 is out of bounds for array of dimension 0
        if len(input.shape) <= 1:
            norms = input
        else:
            # norms = ivy.broadcast_to(
            # ivy.vector_norm(input, axis=dim, ord=p), input.shape)
            # true_dim = (dim - 1) % len(input.shape)
            # true_dim = (dim + 1) % len(input.shape)
            true_dim = 1 if dim != 1 else 0
            # hypothesis.note(f"true_dim: {true_dim}")
            norms = ivy.vector_norm(input, axis=true_dim, ord=p, keepdims=True)

        # norms = input if len(input.shape) <= 1 else
        # ivy.vector_norm(input, axis=dim, ord=p)
        # a, b[:, np.newaxis]
        # if norms.shape != input.shape:
        # if norms.transpose().shape == input.shape:
        #    norms = norms.transpose()

        # else:
        # norms = ivy.repeat(norms, repeats=input.shape[dim-1],
        # axis=dim)
        # repeats_numerator = input.shape[dim-1]
        # repeats_numerator = input.shape[dim]
        # repeats_denominator = norms.shape[dim-1]
        # repeats_denominator = norms.shape[dim]
        # norms = ivy.repeat(norms, repeats=repeats_numerator /
        # repeats_denominator)

        # norms = ivy.broadcast_to(norms, input.shape)
        # norms = norms.transpose()
        # input = input.transpose()
        # norms = ivy.broadcast_to(norms, input.shape)
        # norms = norms.transpose()
        # input = input.transpose()
        # if dim == 0:
        #    if input.shape[0] == 1:
        # norms = norms[ivy.newaxis, ...]
        #    else:
        #        norms = norms[..., ivy.newaxis]
        # norms = norms[..., ivy.newaxis]
        #    norms = ivy.swapaxes(norms, -1, dim)

        if ivy.all(norms == 0):
            ret = ivy.zeros(input.shape, dtype=input_dtype)
        else:
            # hypothesis.note(f"Note, norms : {norms}")
            multiplier = ivy.minimum(maxnorm / norms, ivy.ones_like(norms))
            ret = ivy.multiply(input, multiplier)
            ret = ivy.astype(ret, input_dtype)
        # ivy.prod()

        # ret = ivy.astype(input * norms / maxnorm, dtype=input.dtype)

    # ret
    ret = ivy.astype(ret, input_dtype)
    if ivy.exists(out):
        ivy.inplace_update(input, ret)
    return ret
