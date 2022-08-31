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
    if row == 0 or col == 0:
        return ivy.array([], dtype=dtype)

    lowest_included_diagonal = [
        [i - offset, i] for i in range(-abs(offset), max(row, col) + abs(offset))
    ]

    all_indices = [
        (i, index[1])
        for i in range(
            0, lowest_included_diagonal[-1][1] + 1
        )  # The largest possible value is in the last item, and we wish to include it.
        for index in lowest_included_diagonal
        if 0 <= i < row and 0 <= index[1] < col and i <= index[0]
    ]

    if len(all_indices) == 0:
        return ivy.array([], dtype=dtype)

    print(f"Row: {row}, Col: {col}, Offset: {offset}")

    data = ivy.asarray(all_indices, copy=False, dtype=dtype)
    return data.matrix_transpose()
