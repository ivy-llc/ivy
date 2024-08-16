# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(batch1)) != 3 or len(ivy.shape(batch2)) != 3:
        raise RuntimeError("input must be 3D matrices")
    batch1, batch2 = torch_frontend.promote_types_of_torch_inputs(batch1, batch2)
    ret = ivy.matmul(batch1, batch2, out=out)
    ret = ivy.sum(ret, axis=0, keepdims=False, dtype=ivy.dtype(ret), out=out)
    alpha, ret = torch_frontend.promote_types_of_torch_inputs(alpha, ret)
    ret = ivy.multiply(alpha, ret, out=out)
    beta, input = torch_frontend.promote_types_of_torch_inputs(beta, input)
    beta_input = ivy.multiply(beta, input, out=out)
    beta_input, ret = torch_frontend.promote_types_of_torch_inputs(beta_input, ret)
    return ivy.add(beta_input, ret, out=out)


@to_ivy_arrays_and_back
def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(mat1)) != 2 or len(ivy.shape(mat2)) != 2:
        raise RuntimeError("input must be 2D matrices")
    mat1, mat2 = torch_frontend.promote_types_of_torch_inputs(mat1, mat2)
    ret = ivy.matmul(mat1, mat2, out=out)
    alpha, ret = torch_frontend.promote_types_of_torch_inputs(alpha, ret)
    ret = ivy.multiply(alpha, ret, out=out)
    beta, input = torch_frontend.promote_types_of_torch_inputs(beta, input)
    beta_input = ivy.multiply(beta, input, out=out)
    beta_input, ret = torch_frontend.promote_types_of_torch_inputs(beta_input, ret)
    return ivy.add(beta_input, ret, out=out)


@to_ivy_arrays_and_back
def addmv(input, mat, vec, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(mat)) != 2 or len(ivy.shape(vec)) != 1:
        raise RuntimeError("input must be 2D matrix and 1D vector")
    mat, vec = torch_frontend.promote_types_of_torch_inputs(mat, vec)
    ret = ivy.matmul(mat, vec, out=out)
    alpha, ret = torch_frontend.promote_types_of_torch_inputs(alpha, ret)
    ret = ivy.multiply(alpha, ret, out=out)
    beta, input = torch_frontend.promote_types_of_torch_inputs(beta, input)
    beta_input = ivy.multiply(beta, input, out=out)
    beta_input, ret = torch_frontend.promote_types_of_torch_inputs(beta_input, ret)
    return ivy.add(beta_input, ret, out=out)


@to_ivy_arrays_and_back
def addr(input, vec1, vec2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(vec1)) != 1 or len(ivy.shape(vec2)) != 1:
        raise RuntimeError("input must be 1D vectors")
    vec1, vec2 = torch_frontend.promote_types_of_torch_inputs(vec1, vec2)
    ret = ivy.outer(vec1, vec2, out=out)
    alpha, ret = torch_frontend.promote_types_of_torch_inputs(alpha, ret)
    ret = ivy.multiply(alpha, ret, out=out)
    beta, input = torch_frontend.promote_types_of_torch_inputs(beta, input)
    beta_input = ivy.multiply(beta, input, out=out)
    beta_input, ret = torch_frontend.promote_types_of_torch_inputs(beta_input, ret)
    return ivy.add(beta_input, ret, out=out)


@to_ivy_arrays_and_back
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    if len(ivy.shape(batch1)) != 3 or len(ivy.shape(batch2)) != 3:
        raise RuntimeError("input must be batched 2D matrices")
    batch1, batch2 = torch_frontend.promote_types_of_torch_inputs(batch1, batch2)
    ret = ivy.matmul(batch1, batch2, out=out)
    alpha, ret = torch_frontend.promote_types_of_torch_inputs(alpha, ret)
    ret = ivy.multiply(alpha, ret, out=out)
    beta, input = torch_frontend.promote_types_of_torch_inputs(beta, input)
    beta_input = ivy.multiply(beta, input, out=out)
    beta_input, ret = torch_frontend.promote_types_of_torch_inputs(beta_input, ret)
    return ivy.add(beta_input, ret, out=out)


@to_ivy_arrays_and_back
def bmm(input, mat2, *, out=None):
    if len(ivy.shape(input)) != 3 or len(ivy.shape(mat2)) != 3:
        raise RuntimeError("input must be 3D matrices")
    input, mat2 = torch_frontend.promote_types_of_torch_inputs(input, mat2)
    return ivy.matmul(input, mat2, out=out)


@to_ivy_arrays_and_back
def chain_matmul(*matrices, out=None):
    return ivy.multi_dot(matrices, out=out)


@to_ivy_arrays_and_back
def cholesky(input, upper=False, *, out=None):
    return ivy.cholesky(input, upper=upper, out=out)


@to_ivy_arrays_and_back
def det(input):
    return torch_frontend.linalg.det(input)


@to_ivy_arrays_and_back
def dot(input, other, *, out=None):
    if len(input.shape) == 1 and len(other.shape) == 1:
        input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
        return ivy.matmul(input, other, out=out)
    else:
        raise RuntimeError("input must be 1D vectors")


@to_ivy_arrays_and_back
def ger(input, vec2, *, out=None):
    input, vec2 = torch_frontend.promote_types_of_torch_inputs(input, vec2)
    return ivy.outer(input, vec2, out=out)


@to_ivy_arrays_and_back
def inner(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.inner(input, other, out=out)


@to_ivy_arrays_and_back
def inverse(input, *, out=None):
    return torch_frontend.linalg.inv(input, out=out)


@to_ivy_arrays_and_back
def logdet(input):
    return ivy.det(input).log()


@to_ivy_arrays_and_back
def matmul(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.matmul(input, other, out=out)


@to_ivy_arrays_and_back
def matrix_power(A, n, *, out=None):
    return torch_frontend.linalg.matrix_power(A, n, out=out)


@to_ivy_arrays_and_back
def matrix_rank(input, tol=None, symmetric=False, *, out=None):
    return ivy.matrix_rank(input, atol=tol, hermitian=symmetric, out=out)


@to_ivy_arrays_and_back
def mm(input, mat2, *, out=None):
    if len(ivy.shape(input)) != 2 or len(ivy.shape(mat2)) != 2:
        raise RuntimeError("input must be 2D matrices")
    input, mat2 = torch_frontend.promote_types_of_torch_inputs(input, mat2)
    return ivy.matmul(input, mat2, out=out)


@to_ivy_arrays_and_back
def mv(input, vec, *, out=None):
    if len(ivy.shape(input)) != 2 or len(ivy.shape(vec)) != 1:
        raise RuntimeError("input must be 2D matrix and 1D vector")
    input, vec = torch_frontend.promote_types_of_torch_inputs(input, vec)
    return ivy.matmul(input, vec, out=out)


@to_ivy_arrays_and_back
def outer(input, vec2, *, out=None):
    input, vec2 = torch_frontend.promote_types_of_torch_inputs(input, vec2)
    return ivy.outer(input, vec2, out=out)


@to_ivy_arrays_and_back
def pinverse(input, rcond=1e-15):
    return ivy.pinv(input, rtol=rcond)


@to_ivy_arrays_and_back
def qr(input, some=True, *, out=None):
    if some:
        ret = ivy.qr(input, mode="reduced")
    else:
        ret = ivy.qr(input, mode="complete")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_ivy_arrays_and_back
def slogdet(A, *, out=None):
    return torch_frontend.linalg.slogdet(A, out=out)


@to_ivy_arrays_and_back
def svd(input, some=True, compute_uv=True, *, out=None):
    # TODO: add compute_uv
    if some:
        ret = ivy.svd(input, full_matrices=False)
    else:
        ret = ivy.svd(input, full_matrices=True)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def trapezoid(y, x=None, *, dx=None, dim=-1):
    if x is not None:
        y, x = torch_frontend.promote_types_of_torch_inputs(y, x)
    return ivy.trapz(y, x=x, dx=dx, axis=dim)


@to_ivy_arrays_and_back
def vdot(input, other, *, out=None):
    if len(input.shape) != 1 or len(other.shape) != 1:
        raise RuntimeError("input must be 1D vectors")
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    ret = ivy.vecdot(input, other, out=out)
    return ret.squeeze(0) if ret.ndim == 1 else ret


# alias to fix mm transpilation issue as mm() gets mapped to spmm() after transpilation
spmm = mm
