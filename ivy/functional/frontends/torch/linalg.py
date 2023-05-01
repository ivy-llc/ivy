# local
import math
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes


@to_ivy_arrays_and_back
def vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    return ivy.vector_norm(
        input, axis=dim, keepdims=keepdim, ord=ord, out=out, dtype=dtype
    )


@to_ivy_arrays_and_back
def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
    return torch_frontend.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)


@to_ivy_arrays_and_back
def divide(input, other, *, rounding_mode=None, out=None):
    return ivy.divide(input, other, out=out)


@to_ivy_arrays_and_back
def inv(input, *, out=None):
    return ivy.inv(input, out=out)


@to_ivy_arrays_and_back
def pinv(input, *, atol=None, rtol=None, hermitian=False, out=None):
    # TODO: add handling for hermitian once complex numbers are supported
    if atol is None:
        return ivy.pinv(input, rtol=rtol, out=out)
    else:
        sigma = ivy.svdvals(input)[0]
        if rtol is None:
            rtol = atol / sigma
        else:
            if atol > rtol * sigma:
                rtol = atol / sigma

    return ivy.pinv(input, rtol=rtol, out=out)


@to_ivy_arrays_and_back
def det(input, *, out=None):
    return ivy.det(input, out=out)


@to_ivy_arrays_and_back
def eigvals(input, *, out=None):
    # TODO: add handling for out
    return ivy.eigvals(input)


@to_ivy_arrays_and_back
def eigvalsh(input, UPLO="L", *, out=None):
    return ivy.eigvalsh(input, UPLO=UPLO, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def eigh(a, /, UPLO="L", out=None):
    return ivy.eigh(a, UPLO=UPLO, out=out)


@to_ivy_arrays_and_back
def qr(input, mode="reduced", *, out=None):
    if mode == "reduced":
        ret = ivy.qr(input, mode="reduced")
    elif mode == "r":
        Q, R = ivy.qr(input, mode="r")
        Q = []
        ret = Q, R
    elif mode == "complete":
        ret = ivy.qr(input, mode="complete")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_ivy_arrays_and_back
def slogdet(input, *, out=None):
    # TODO: add handling for out
    return ivy.slogdet(input)


@with_unsupported_dtypes({"2.0.0 and below": ("bfloat16", "float16")}, "torch")
@to_ivy_arrays_and_back
def cond(input, p=None, *, out=None):
    return ivy.cond(input, p=p, out=out)


@to_ivy_arrays_and_back
def matrix_power(input, n, *, out=None):
    return ivy.matrix_power(input, n, out=out)


@with_supported_dtypes(
    {"1.11.0 and below": ("float32", "float64", "complex64", "complex128")}, "torch"
)
@to_ivy_arrays_and_back
def matrix_norm(input, ord="fro", dim=(-2, -1), keepdim=False, *, dtype=None, out=None):
    if "complex" in ivy.as_ivy_dtype(input.dtype):
        input = ivy.abs(input)
    if dtype:
        input = ivy.astype(input, dtype)
    return ivy.matrix_norm(input, ord=ord, axis=dim, keepdims=keepdim, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def cross(input, other, *, dim=None, out=None):
    return torch_frontend.miscellaneous_ops.cross(input, other, dim=dim, out=out)


@to_ivy_arrays_and_back
def vecdot(x1, x2, *, dim=-1, out=None):
    return ivy.vecdot(x1, x2, axis=dim, out=out)


@to_ivy_arrays_and_back
def matrix_rank(input, *, atol=None, rtol=None, hermitian=False, out=None):
    # TODO: add handling for hermitian once complex numbers are supported
    return ivy.astype(ivy.matrix_rank(input, atol=atol, rtol=rtol, out=out), ivy.int64)


@to_ivy_arrays_and_back
def cholesky(input, *, upper=False, out=None):
    return ivy.cholesky(input, upper=upper, out=out)


@to_ivy_arrays_and_back
def svd(A, /, *, full_matrices=True, driver=None, out=None):
    # TODO: add handling for driver and out
    return ivy.svd(A, compute_uv=True, full_matrices=full_matrices)


@to_ivy_arrays_and_back
def svdvals(A, *, driver=None, out=None):
    # TODO: add handling for driver
    return ivy.svdvals(A, out=out)


@to_ivy_arrays_and_back
def inv_ex(input, *, check_errors=False, out=None):
    try:
        inputInv = ivy.inv(input, out=out)
        info = ivy.zeros(input.shape[:-2], dtype=ivy.int32)
        return inputInv, info
    except RuntimeError as e:
        if check_errors:
            raise RuntimeError(e)
        else:
            inputInv = input * math.nan
            info = ivy.ones(input.shape[:-2], dtype=ivy.int32)
            return inputInv, info


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
def tensorinv(input, ind=2, *, out=None):
    not_invertible = "Reshaped tensor is not invertible"
    prod_cond = "Tensor shape must satisfy prod(A.shape[:ind]) == prod(A.shape[ind:])"
    positive_ind_cond = "Expected a strictly positive integer for 'ind'"
    input_shape = ivy.shape(input)
    assert ind > 0, f"{positive_ind_cond}"
    shape_ind_end = input_shape[:ind]
    shape_ind_start = input_shape[ind:]
    prod_ind_end = 1
    prod_ind_start = 1
    for i in shape_ind_start:
        prod_ind_start *= i
    for j in shape_ind_end:
        prod_ind_end *= j
    assert prod_ind_end == prod_ind_start, f"{prod_cond}."
    inverse_shape = shape_ind_start + shape_ind_end
    input = ivy.reshape(input, shape=(prod_ind_end, -1))
    inverse_shape_tuple = tuple([*inverse_shape])
    assert inv_ex(input, check_errors=True), f"{not_invertible}."
    inverse_tensor = ivy.inv(input)
    return ivy.reshape(inverse_tensor, shape=inverse_shape_tuple, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def eig(input, *, out=None):
    return ivy.eig(input, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def solve(input, other, *, out=None):
    return ivy.solve(input, other, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def tensorsolve(A, B, dims=None, *, out=None):
    return ivy.tensorsolve(A, B, axes=dims, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def lu_factor(A, *, pivot=True, out=None):
    return ivy.lu_factor(A, pivot=pivot, out=out)


@to_ivy_arrays_and_back
def matmul(input, other, *, out=None):
    return ivy.matmul(input, other, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def vander(x, N=None):
    if len(x.shape) < 1:
        raise RuntimeError("Input dim must be greater than or equal to 1.")

    # pytorch always return int64 for integers
    if "int" in x.dtype:
        x = ivy.astype(x, ivy.int64)

    if len(x.shape) == 1:
        # torch always returns the powers in ascending order
        return ivy.vander(x, N=N, increasing=True)

    # support multi-dimensional array
    original_shape = x.shape
    if N is None:
        N = x.shape[-1]

    # store the vander output
    x = ivy.reshape(x, (-1, x.shape[-1]))
    output = []

    for i in range(x.shape[0]):
        output.append(ivy.vander(x[i], N=N, increasing=True))

    output = ivy.stack(output)
    output = ivy.reshape(output, (*original_shape, N))
    output = ivy.astype(output, x.dtype)
    return output


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def multi_dot(tensors, *, out=None):
    return ivy.multi_dot(tensors, out=out)
