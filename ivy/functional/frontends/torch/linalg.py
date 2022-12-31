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
def eigvalsh(input, UPLO="L", *, out=None):
    return ivy.eigvalsh(input, UPLO=UPLO, out=out)


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
def cross(input, other, *, dim=-1, out=None):
    return torch_frontend.cross(input, other, dim, out=out)


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


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
def eig(input, *, out=None):
    return ivy.eig(input, out=out)
