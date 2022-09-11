# global
import ivy


def addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(batch1)) != 3 or len(ivy.shape(batch2)) != 3:
        raise RuntimeError("input must be 3D matrices")
    ret = ivy.matmul(batch1, batch2, out=out)
    ret = ivy.sum(ret, axis=0, keepdims=False, dtype=ivy.dtype(ret), out=out)
    ret = ivy.multiply(alpha, ret, out=out)
    return ivy.add(ivy.multiply(beta, input, out=out), ret, out=out)


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(mat1)) != 2 or len(ivy.shape(mat2)) != 2:
        raise RuntimeError("input must be 2D matrices")
    ret = ivy.matmul(mat1, mat2, out=out)
    ret = ivy.multiply(alpha, ret, out=out)
    return ivy.add(ivy.multiply(beta, input, out=out), ret, out=out)


def addmv(input, mat, vec, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(mat)) != 2 or len(ivy.shape(vec)) != 1:
        raise RuntimeError("input must be 2D matrix and 1D vector")
    ret = ivy.matmul(mat, vec, out=out)
    ret = ivy.multiply(alpha, ret, out=out)
    return ivy.add(ivy.multiply(beta, input, out=out), ret, out=out)


def addr(input, vec1, vec2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(vec1)) != 1 or len(ivy.shape(vec2)) != 1:
        raise RuntimeError("input must be 1D vectors")
    ret = ivy.outer(vec1, vec2, out=out)
    ret = ivy.multiply(alpha, ret, out=out)
    return ivy.add(ivy.multiply(beta, input, out=out), ret, out=out)


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(batch1)) != 3 or len(ivy.shape(batch2)) != 3:
        raise RuntimeError("input must be batched 2D matrices")
    ret = ivy.matmul(batch1, batch2, out=out)
    ret = ivy.multiply(alpha, ret, out=out)
    return ivy.add(ivy.multiply(beta, input, out=out), ret, out=out)


def bmm(input, mat2, *, out=None):
    if len(ivy.shape(input)) != 3 or len(ivy.shape(mat2)) != 3:
        raise RuntimeError("input must be 3D matrices")
    return ivy.matmul(input, mat2, out=out)


def cholesky(input, upper=False, *, out=None):
    return ivy.cholesky(input, upper=upper, out=out)


def ger(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


def inverse(input, *, out=None):
    return ivy.inv(input, out=out)


def det(input):
    return ivy.det(input)


def logdet(input):
    return ivy.det(input).log()


def slogdet(input):
    return ivy.slogdet(input)


def matmul(input, other, *, out=None):
    return ivy.matmul(input, other, out=out)


def matrix_power(input, n, *, out=None):
    return ivy.matrix_power(input, n, out=out)


def matrix_rank(input, tol=None, symmetric=False, *, out=None):
    # TODO: add symmetric
    return ivy.matrix_rank(input, rtol=tol, out=out).astype("int64")


def mm(input, mat2, *, out=None):
    if len(ivy.shape(input)) != 2 or len(ivy.shape(mat2)) != 2:
        raise RuntimeError("input must be 2D matrices")
    return ivy.matmul(input, mat2, out=out)


def mv(input, vec, *, out=None):
    if len(ivy.shape(input)) != 2 or len(ivy.shape(vec)) != 1:
        raise RuntimeError("input must be 2D matrix and 1D vector")
    return ivy.matmul(input, vec, out=out)


def outer(input, vec2, *, out=None):
    return ivy.outer(input, vec2, out=out)


def pinverse(input, rcond=1e-15):
    return ivy.pinv(input, rtol=rcond)


def qr(input, some=True, *, out=None):
    if some:
        ret = ivy.qr(input, mode="reduced")
    else:
        ret = ivy.qr(input, mode="complete")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def svd(input, some=True, compute_uv=True, *, out=None):
    # TODO: add compute_uv
    if some:
        ret = ivy.svd(input, full_matrices=False)
    else:
        ret = ivy.svd(input, full_matrices=True)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def vdot(input, other, *, out=None):
    if len(ivy.shape(input)) != 1 or len(ivy.shape(other)) != 1:
        raise RuntimeError("input must be 1D vectors")
    return ivy.vecdot(input, other, out=out)
