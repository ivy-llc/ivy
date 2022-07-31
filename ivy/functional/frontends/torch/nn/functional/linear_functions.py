import ivy


def linear(input, weight, bias=None):
    return ivy.add(ivy.matmul(input, weight), ivy.array([bias]))
