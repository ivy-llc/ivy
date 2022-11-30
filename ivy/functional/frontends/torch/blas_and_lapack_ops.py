# global
import ivy
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
