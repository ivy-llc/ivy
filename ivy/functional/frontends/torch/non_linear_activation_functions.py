# global
import ivy


def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)


def leaky_relu(input, negative_slope=0.01):
    return ivy.leaky_relu(input, negative_slope)


leaky_relu.unsupported_dtypes = ("float16",)
