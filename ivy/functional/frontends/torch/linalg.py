# global
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def cholesky(input, upper=False, *, out=None):
    return ivy.cholesky(input, upper=upper, out=out)


@to_ivy_arrays_and_back
def det(input):
    return ivy.det(input)


@to_ivy_arrays_and_back
def ger(input, vec2, *, out=None):
    input, vec2 = torch_frontend.promote_types_of_torch_inputs(input, vec2)
    return ivy.outer(input, vec2, out=out)


@to_ivy_arrays_and_back
def inverse(input, *, out=None):
    return ivy.inv(input, out=out)


@to_ivy_arrays_and_back
def logdet(input):
    return ivy.det(input).log()


@to_ivy_arrays_and_back
def matmul(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.matmul(input, other, out=out)


@to_ivy_arrays_and_back
def matrix_power(input, n, *, out=None):
    return ivy.matrix_power(input, n, out=out)


@to_ivy_arrays_and_back
def matrix_rank(input, tol=None, symmetric=False, *, out=None):
    # TODO: add symmetric
    return ivy.matrix_rank(input, rtol=tol, out=out).astype("int64")


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
def slogdet(input):
    return ivy.slogdet(input)


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


@to_ivy_arrays_and_back
def vdot(input, other, *, out=None):
    if len(ivy.shape(input)) != 1 or len(ivy.shape(other)) != 1:
        raise RuntimeError("input must be 1D vectors")
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.vecdot(input, other, out=out)
