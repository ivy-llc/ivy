# global
import ivy

# matrix_power


def matrix_power(a, n):
    return ivy.matrix_power(a, n)


matrix_power.unsupported_dtypes = ("float16",)
