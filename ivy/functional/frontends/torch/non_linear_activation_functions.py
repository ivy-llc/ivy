# global
import ivy


def sigmoid(input, out=None):
    return ivy.sigmoid(input, out=out)


sigmoid.unsupported_dtypes = ("float16",)


def softmax(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


softmax.unsupported_dtypes = ("float16",)

def gelu(input):
    return ivy.gelu(input)

gelu.unsupported_dtypes = ("float16",)
