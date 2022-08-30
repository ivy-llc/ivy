# global
import ivy


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(mat1)) != 2 or len(ivy.shape(mat2)) != 2:
        raise RuntimeError("input must be matrices")
    ret = ivy.matmul(mat1, mat2, out=out)
    ret = ivy.multiply(alpha, ret, out=out)
    return ivy.add(ivy.multiply(beta, input, out=out), ret, out=out)


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
    if len(ivy.shape(input)) != 2 or len(ivy.shape(mat2)) != 2:
        raise RuntimeError("input must be matrices")
    return ivy.matmul(input, mat2, out=out)


def outer(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


outer.unsupported_dtypes = {"numpy": ("float16", "int8")}


def pinverse(input, rcond=1e-15):
    return ivy.pinv(input, rcond)


pinverse.unsupported_dtypes = ("float16",)


def qr(input, some=True, *, out=None):
    if some:
        ret = ivy.qr(input, mode="reduced")
    else:
        ret = ivy.qr(input, mode="complete")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


qr.unsupported_dtypes = ("float16",)


def svd(input, some=True, compute_uv=True, *, out=None):
    # TODO: add compute_uv checks
    if some:
        ret = ivy.svd(input, full_matrices=False)
    else:
        ret = ivy.svd(input, full_matrices=True)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


svd.unsupported_dtypes = ("float16",)
