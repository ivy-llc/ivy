# global
import ivy


def hard_sigmoid(x):
    x = ivy.hard_sigmoid(x)
    return x


fill.unsupported_dtypes = {"torch": "float16"}
