# local
import math
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from collections import namedtuple


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
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
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def inv(A, *, out=None):
    return ivy.inv(A, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def pinv(input, *, atol=None, rtol=None, hermitian=False, out=None):
    # TODO: add handling for hermitian
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
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def det(A, *, out=None):
    return ivy.det(A, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def eigvals(input, *, out=None):
    ret = ivy.eigvals(input)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def eigvalsh(input, UPLO="L", *, out=None):
    ret = ivy.eigvalsh(input, UPLO=UPLO, out=out)
    if "complex64" in ivy.as_ivy_dtype(ret.dtype):
        ret = ivy.astype(ret, ivy.float32)
    elif "complex128" in ivy.as_ivy_dtype(ret.dtype):
        ret = ivy.astype(ret, ivy.float64)
    return ret


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def eigh(a, /, UPLO="L", out=None):
    return ivy.eigh(a, UPLO=UPLO, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def qr(A, mode="reduced", *, out=None):
    if mode == "reduced":
        ret = ivy.qr(A, mode="reduced")
    elif mode == "r":
        Q, R = ivy.qr(A, mode="r")
        Q = []
        ret = Q, R
    elif mode == "complete":
        ret = ivy.qr(A, mode="complete")
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def slogdet(A, *, out=None):
    sign, logabsdet = ivy.slogdet(A)
    if "complex64" in ivy.as_ivy_dtype(logabsdet.dtype):
        logabsdet = ivy.astype(logabsdet, ivy.float32)
    if "complex128" in ivy.as_ivy_dtype(logabsdet.dtype):
        logabsdet = ivy.astype(logabsdet, ivy.float64)
    ret = namedtuple("slogdet", ["sign", "logabsdet"])(sign, logabsdet)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret, keep_input_dtype=True)
    return ret


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.0.1 and below": ("float32", "float64", "complex")}, "torch")
def cond(input, p=None, *, out=None):
    return ivy.cond(input, p=p, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matrix_power(A, n, *, out=None):
    return ivy.matrix_power(A, n, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.0.1 and below": ("float32", "float64", "complex")}, "torch")
def matrix_exp(A):
    return ivy.matrix_exp(A)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matrix_norm(input, ord="fro", dim=(-2, -1), keepdim=False, *, dtype=None, out=None):
    if "complex" in ivy.as_ivy_dtype(input.dtype):
        input = ivy.abs(input)
    if dtype:
        input = ivy.astype(input, ivy.as_ivy_dtype(dtype))
    return ivy.matrix_norm(input, ord=ord, axis=dim, keepdims=keepdim, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def cross(input, other, *, dim=None, out=None):
    return torch_frontend.miscellaneous_ops.cross(input, other, dim=dim, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def vecdot(x, y, *, dim=-1, out=None):
    if "complex" in ivy.as_ivy_dtype(x.dtype):
        x = ivy.conj(x)
    return ivy.sum(ivy.multiply(x, y), axis=dim)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
    return ivy.matrix_rank(A, atol=atol, rtol=rtol, hermitian=hermitian, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def cholesky(input, *, upper=False, out=None):
    return ivy.cholesky(input, upper=upper, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def svd(A, /, *, full_matrices=True, driver=None, out=None):
    # TODO: add handling for driver and out
    return ivy.svd(A, compute_uv=True, full_matrices=full_matrices)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def svdvals(A, *, driver=None, out=None):
    # TODO: add handling for driver
    return ivy.svdvals(A, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def inv_ex(A, *, check_errors=False, out=None):
    if ivy.any(ivy.det(A) == 0):
        if check_errors:
            raise RuntimeError("Singular Matrix")
        else:
            inv = A * math.nan
            # TODO: info should return an array containing the diagonal element of the
            # LU decomposition of the input matrix that is exactly zero
            info = ivy.ones(A.shape[:-2], dtype=ivy.int32)
    else:
        inv = ivy.inv(A, out=out)
        info = ivy.zeros(A.shape[:-2], dtype=ivy.int32)
    return inv, info


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
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


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, "torch")
def eig(input, *, out=None):
    return ivy.eig(input, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def solve(A, B, *, left=True, out=None):
    # TODO: Implement left
    return ivy.solve(A, B, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def tensorsolve(A, B, dims=None, *, out=None):
    return ivy.tensorsolve(A, B, axes=dims, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def lu_factor(A, *, pivot=True, out=None):
    return ivy.lu_factor(A, pivot=pivot, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matmul(input, other, *, out=None):
    return ivy.matmul(input, other, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.0.1 and below": ("integer", "float", "complex")}, "torch")
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
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def multi_dot(tensors, *, out=None):
    return ivy.multi_dot(tensors, out=out)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def solve_ex(A, B, *, left=True, check_errors=False, out=None):
    # TODO: Implement left
    try:
        result = ivy.solve(A, B, out=out)
        info = ivy.zeros(A.shape[:-2], dtype=ivy.int32)
        return result, info
    except RuntimeError as e:
        if check_errors:
            raise RuntimeError(e)
        else:
            result = A * math.nan
            info = ivy.ones(A.shape[:-2], dtype=ivy.int32)

            return result, info


@to_ivy_arrays_and_back
def cholesky_ex(input, *, upper=False, check_errors=False, out=None):
    try:
        matrix = ivy.cholesky(input, upper=upper, out=out)
        info = ivy.zeros(input.shape[:-2], dtype=ivy.int32)
        return matrix, info
    except RuntimeError as e:
        if check_errors:
            raise RuntimeError(e)
        else:
            matrix = input * math.nan
            info = ivy.ones(input.shape[:-2], dtype=ivy.int32)
            return matrix, info


def lu_factor_ex(A, *, pivot=True, check_errors=False, out=None):
    try:
        LU, pivots = ivy.lu_factor(A, pivot=pivot, out=out)
        info = ivy.zeros(A.shape[:2], dtype=ivy.int32)
        return LU, pivots, info
    except RuntimeError as e:
        if check_errors:
            raise RuntimeError(e)
        else:
            LU = A * math.nan
            info = ivy.ones(A.shape[:2], dtype=ivy.int32)
