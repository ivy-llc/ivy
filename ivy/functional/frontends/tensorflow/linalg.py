import ivy


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("float16", "bfloat16")


def trace(input, offset):
    return ivy.trace(input, offset)


det.unsupported_dtypes = ("float16", "bfloat16")
