# global
import ivy


def solve(a, b, out=None):
    return ivy.solve(a, b, out=None)


solve.unsupported_dtypes = ("float16",)
