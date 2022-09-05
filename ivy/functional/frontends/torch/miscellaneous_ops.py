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


def tril_indices(row, col, offset=0, *, dtype="int64", device="cpu", layout=None):
    if row == 0 and col == 0:
        return ivy.array([], dtype=dtype, device=device)

    highest_included_diagonal = [
        [i - offset, i] for i in range(-abs(offset), max(row, col) + abs(offset))
    ]

    needs_transposition = True

    if row == 0:
        if col == 1:
            all_indices = [[0, 0]]
        elif offset < col:
            all_indices = [[], []]
            needs_transposition = False
        else:
            all_indices = [[0, i] for i in range(min(col, offset))]

    else:
        all_indices = [
            (i, index[1])
            for i in range(
                highest_included_diagonal[0][1], max(row, col)
            )  # The smallest possible value is in the first item
            for index in highest_included_diagonal
            if 0 <= i < row and 0 <= index[1] < col and i >= index[0]
        ]

    if len(all_indices) == 0:
        return ivy.array([], dtype=dtype, device=device)

    data = ivy.asarray(all_indices, copy=False, dtype=dtype)
    if needs_transposition:
        data = data.matrix_transpose()
    return data
