import ivy


def hard_sigmoid(x, name=None):
    return ivy.hard_sigmoid(x)


hard_sigmoid.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def linear(x, y, name=None):
    return ivy.linear(x, y)


linear.unsupported_dtypes = {"torch": ("float16", "bfloat16")}