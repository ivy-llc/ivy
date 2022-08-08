import ivy


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("bfloat16",)
