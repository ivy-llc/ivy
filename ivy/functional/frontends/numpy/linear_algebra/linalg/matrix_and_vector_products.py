# local
import ivy


# matmul
def matmul(
    x1, x2, /, out=None, *, casting="same_kind", order="K", dtype=None, subok=True
):
    return ivy.matmul(x1, x2, out=out)


# matrix_power
def matrix_power(a, n):
    return ivy.matrix_power(a, n)
