# global
import ivy

def leaky_relu():
    return ivy.leaky_rely()

def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)


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
