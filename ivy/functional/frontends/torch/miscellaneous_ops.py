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


def triu_indices(row, col, offset=0, dtype="int64", device="cpu", layout=None):
    lowest_included_diagonal = [
        [i, i - offset] for i in range(-abs(offset), max(row, col) + abs(offset))
    ]

    # all_indices = []
    all_indices = [
        [i, index[1]]
        for index in lowest_included_diagonal
        for i in range(0, index[0] + 1)
    ]
    # for index in lowest_included_diagonal:
    # [X, increasing_number] until intersects lowest_included_diagonal
    # all_indices.append([index[0], i] for i in range(0, index[1]+1))
    #    all_indices += [[i, index[1]] for i in range(0, index[0] + 1)]

    data = ivy.asarray(all_indices, copy=False)
    return data.matrix_transpose()
    # return ivy.array(all_indices).matrix_transpose()
