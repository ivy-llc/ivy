# local
import ivy


def linear(input, weight, bias=None):
    return ivy.linear(input, weight, bias=bias)
