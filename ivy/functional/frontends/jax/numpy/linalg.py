# local
import ivy
from ivy.functional.frontends.jax import DeviceArray
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.jax.numpy import promote_types_of_jax_inputs


@to_ivy_arrays_and_back
def inv(a):
    return ivy.inv(a)


@to_ivy_arrays_and_back
def det(a):
    return ivy.det(a)


@to_ivy_arrays_and_back
def svd(a, /, *, full_matrices=True, compute_uv=True, hermitian=None):
    if not compute_uv:
        return ivy.svdvals(a)
    return ivy.svd(a, full_matrices=full_matrices)


@to_ivy_arrays_and_back
def eig(a):
    return ivy.eig(a)


@to_ivy_arrays_and_back
def eigh(a, UPLO="L", symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        a = symmetrize(a)

    return ivy.eigh(a, UPLO=UPLO)


@to_ivy_arrays_and_back
def eigvalsh(a, UPLO="L"):
    return ivy.eigvalsh(a, UPLO=UPLO)


@to_ivy_arrays_and_back
def qr(a, mode="reduced"):
    return ivy.qr(a, mode=mode)


@to_ivy_arrays_and_back
def eigvals(a):
    return ivy.eigh(a)


@to_ivy_arrays_and_back
def cholesky(a):
    return ivy.cholesky(a)


@to_ivy_arrays_and_back
def slogdet(a, method=None):
    return ivy.slogdet(a)


@to_ivy_arrays_and_back
def matrix_rank(M):
    return ivy.matrix_rank(M)


@to_ivy_arrays_and_back
def solve(a, b):
    return ivy.solve(a, b)


@to_ivy_arrays_and_back
def pinv(a, rcond=None):
    return ivy.pinv(a, rtol=rcond)


@to_ivy_arrays_and_back
def norm(x, ord=None, axis=None, keepdims=False):
    if ord is None:
        ord = 2
    if type(axis) in [list, tuple] and len(axis) == 2:
        return DeviceArray(ivy.matrix_norm(x, ord=ord, axis=axis, keepdims=keepdims))
    return DeviceArray(ivy.vector_norm(x, ord=ord, axis=axis, keepdims=keepdims))


norm.supported_dtypes = (
    "float32",
    "float64",
)


@to_ivy_arrays_and_back
def matrix_power(a, n):
    return ivy.matrix_power(a, n)


@to_ivy_arrays_and_back
def tensorsolve(a, b, axes=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.tensorsolve(a, b, axes=axes)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"0.3.14 and below": ("float16", "bfloat16")}, "jax")
def tensorinv(a, ind=2):
    old_shape = ivy.shape(a)
    prod = 1
    if ind > 0:
        invshape = old_shape[ind:] + old_shape[:ind]
        for k in old_shape[ind:]:
            prod *= k
    else:
        raise ValueError("Invalid ind argument.")
    a = ivy.reshape(a, shape=(prod, -1))
    ia = ivy.inv(a)
    new_shape = tuple([*invshape])
    return DeviceArray(ivy.reshape(ia, shape=new_shape))


@to_ivy_arrays_and_back
def cond(x, p=None):
    for a in x:
        if a.size == 0 and ivy.prod(a.shape[-2:]) == 0:
            raise ValueError("Arrays cannot be empty")
    if p in (None, 2):
        s = ivy.svd(x, compute_uv=False)
        return s[..., 0] / s[..., -1]
    elif p == -2:
        s = ivy.svd(x, compute_uv=False)
        r = s[..., -1] / s[..., 0]
    else:
        if ivy.get_num_dims(x) < 2:
            raise ValueError(
                "%d-dimensional array given."
                "Array must be at least two-dimensional" % ivy.get_num_dims(x)
            )
        m, n = ivy.shape(x)[-2:]
        if m != n:
            raise ValueError("Last 2 dimensions of the array must be square")
        invx = ivy.inv(x)
        r = ivy.matrix_norm(x, ord=p, axis=(-2, -1)) * ivy.norm(
            invx, ord=p, axis=(-2, -1)
        )
    # Convert nans to infs unless the original array had nan entries
    orig_nan_check = ivy.full_like(r, ~ivy.isnan(r).any())
    nan_mask = ivy.logical_and(ivy.isnan(r), ~ivy.isnan(x).any(axis=(-2, -1)))
    r = ivy.where(orig_nan_check, ivy.where(nan_mask, ivy.inf, r), r)
    return r
