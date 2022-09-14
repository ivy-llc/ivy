import ivy


def dist(input, other, p=2):
    return ivy.vector_norm(ivy.subtract(input, other), ord = p)
