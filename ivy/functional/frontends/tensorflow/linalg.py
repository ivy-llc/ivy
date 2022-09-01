# global
import ivy


def matrix_rank(a, tol=None, valiate_args=False, name=None):
    return ivy.matrix_rank(a, tol)


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


def pinv(a, rcond=None, validate_args=False, name=None):
    return ivy.pinv(a, rcond)


pinv.unsupported_dtypes = ("float16", "bfloat16")
