# global
import ivy


def cholesky(input, upper=False, *, out=None):
    return ivy.cholesky(input, upper, out=out)


cholesky.unsupported_dtypes = ("float16",)


def ger(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


ger.unsupported_dtypes = {"numpy": ("float16", "int8")}


def inverse(input, *, out=None):
    return ivy.inv(input, out=out)


inverse.unsupported_dtypes = ("float16",)


def det(input):
    return ivy.det(input)


det.unsupported_dtypes = ("float16",)


def slogdet(input):
    return ivy.slogdet(input)


slogdet.unsupported_dtypes = ("float16",)


def matmul(input, other, *, out=None):
    return ivy.matmul(input, other, out=out)


def mm(input, mat2, *, out=None):
    return ivy.matmul(input, mat2, out=out)


def outer(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


outer.unsupported_dtypes = {"numpy": ("float16", "int8")}


def pinverse(input, rcond=1e-15):
    return ivy.pinv(input, rcond)


def qr(input, some=True, *, out=None):
    if some:
        return ivy.qr(input, mode="reduced", out=out)
    return ivy.qr(input, mode="complete", out=out)


def svd(input, some=True, compute_uv=True, *, out=None):
    # TODO: add compute_uv checks
    if some:
        return ivy.svd(input, full_matrices=False, out=out)
    return ivy.svd(input, full_matrices=True, out=out)
