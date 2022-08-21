# global
import ivy


def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)
<<<<<<< HEAD
=======


def leaky_relu(input, negative_slope=0.01):
    return ivy.leaky_relu(input, negative_slope)


leaky_relu.unsupported_dtypes = ("float16",)


def softmax(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


softmax.unsupported_dtypes = ("float16",)


def gelu(input, approximate="none"):
    if approximate == "none":
        approximate = False
    else:
        approximate = True
    return ivy.gelu(input, approximate)


gelu.unsupported_dtypes = ("float16",)
>>>>>>> 241a3c87d774fb0877df3ef70ff67e83a6cbe4be
