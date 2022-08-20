import ivy


def flip(input, dims):
    return ivy.flip(input, dims)


def roll(input, shifts, dims=None):
    return ivy.roll(input, shifts, dims)


def repeat_interleave(input, repeats, dim=None):
    return ivy.repeat(input, repeats, dim=dim)
