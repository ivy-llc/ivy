import ivy


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("float16", "bfloat16")


def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


eigvalsh.unsupported_dtypes = ("float16", "bfloat16")


def solve(x, y):
    return ivy.solve(x, y)


solve.unsupported_dtypes = ("float16", "bfloat16")


def slogdet(input, name=None):
    return ivy.slogdet(input)


slogdet.unsupported_dtypes = ("float16", "bfloat16")


def tensordot(x, y, axes, name=None):
    return ivy.tensordot(x, y, axes)


tensordot.supported_dtypes = ("float32", "float64")
