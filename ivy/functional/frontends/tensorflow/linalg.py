import ivy


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("float16", "bfloat16")


def eigvals(tensor, name=None):
    return ivy.eigvals(tensor)


eigvals.unsupported_dtypes = ("float16", "bfloat16")
