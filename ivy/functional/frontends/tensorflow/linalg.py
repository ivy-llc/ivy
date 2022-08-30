import ivy


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("float16", "bfloat16")


def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


eigvalsh.unsupported_dtypes = ("float16", "bfloat16")


def cholesky_solve(chol, rhs, name=None):
    return ivy.solve(chol, rhs)

cholesky_solve.unsupported_dtypes = ("float16", "bfloat16")