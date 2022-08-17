import ivy


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("float16", "bfloat16")


def eigh(tensor, name=None):
    return ivy.eigh(tensor)


eigh.unsupported_dtypes = ("float16", "bfloat16")