# global
import ivy


def matrix_rank(a, tol=None, valiate_args=False, name=None):
    return ivy.matrix_rank(a, tol)


def det(input, name=None):
    return ivy.det(input)


def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


def solve(x, y):
    return ivy.solve(x, y)


def slogdet(input, name=None):
    return ivy.slogdet(input)


def pinv(a, rcond=None, validate_args=False, name=None):
    return ivy.pinv(a, rcond)


def tensordot(x, y, axes, name=None):
    return ivy.tensordot(x, y, axes)
