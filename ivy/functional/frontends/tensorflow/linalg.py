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

def norm(x, axis=None, keepdims=False, ord=2):
    return ivy.matrix_norm(x, axis, keepdims, ord)

norm.unsupported_dtypes("float16", "bfloat16")