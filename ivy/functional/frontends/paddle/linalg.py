# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle import promote_types_of_paddle_inputs
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes({"2.4.1 and above": ("int64",)}, "paddle")
@to_ivy_arrays_and_back
def bincount(x, weights=None, minlength=0, name=None):
    return ivy.bincount(x, weights=weights, minlength=minlength)


# bmm
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def bmm(x, y, transpose_x=False, transpose_y=False, name=None):
    if len(ivy.shape(x)) != 3 or len(ivy.shape(y)) != 3:
        raise RuntimeError("input must be 3D matrices")
    x, y = promote_types_of_paddle_inputs(x, y)
    return ivy.matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)


# cholesky
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def cholesky(x, /, *, upper=False, name=None):
    return ivy.cholesky(x, upper=upper)


# cholesky_solve
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def cholesky_solve(x, y, /, *, upper=False, name=None):
    if upper:
        y = ivy.matrix_transpose(y)
    Y = ivy.solve(y, x)
    return ivy.solve(ivy.matrix_transpose(y), Y)


# cond
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def cond(x, p=None, name=None):
    ret = ivy.cond(x, p=p, out=name)
    if ret.shape == ():
        ret = ret.reshape((1,))
    return ret


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def cross(x, y, /, *, axis=9, name=None):
    x, y = promote_types_of_paddle_inputs(x, y)
    return ivy.cross(x, y, axis=axis)


# diagonal
@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float64",
            "complex128",
            "float32",
            "complex64",
            "bool",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    return ivy.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


@with_supported_dtypes({"2.4.1 and above": ("float64", "float32")}, "paddle")
@to_ivy_arrays_and_back
def dist(x, y, p=2):
    ret = ivy.vector_norm(ivy.subtract(x, y), ord=p)
    return ivy.reshape(ret, (1,))


# dot
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def dot(x, y, name=None):
    x, y = promote_types_of_paddle_inputs(x, y)
    out = ivy.multiply(x, y)
    return ivy.sum(out, axis=ivy.get_num_dims(x) - 1, keepdims=False)


# eig
@to_ivy_arrays_and_back
def eig(x, name=None):
    return ivy.eig(x)


# eigh
@to_ivy_arrays_and_back
def eigh(x, UPLO="L", name=None):
    return ivy.eigh(x, UPLO=UPLO)


# eigvals
@to_ivy_arrays_and_back
def eigvals(x, name=None):
    return ivy.eigvals(x)


# eigvalsh
@to_ivy_arrays_and_back
def eigvalsh(x, UPLO="L", name=None):
    return ivy.eigvalsh(x, UPLO=UPLO)


@to_ivy_arrays_and_back
def lu_unpack(lu_data, lu_pivots, unpack_datas=True, unpack_pivots=True, *, out=None):
    A = lu_data
    n = A.shape
    m = len(lu_pivots)
    pivot_matrix = ivy.eye(m)
    L = ivy.tril(A)
    L.fill_diagonal(1.000)
    U = ivy.triu(A)
    for i in range(m):
        if i != lu_pivots[i] - 1:
            pivot_matrix[[i, lu_pivots[i] - 1]] = pivot_matrix[[lu_pivots[i] - 1, i]]
        P = pivot_matrix
    if not unpack_datas:
        L = ivy.zeros(n)
        U = ivy.zeros(n)
        if not unpack_pivots:
            P = ivy.zeros(n)
        else:
            P = pivot_matrix
        result = f"P={P}\n" + f"L={L}\n" + f"U={U}"
        return result
    elif not unpack_pivots:
        P = ivy.zeros(n)
        result = f"P={P}\n" + f"L={L}\n" + f"U={U}"
        return result
    else:
        result = f"P={P}\n" + f"L={L}\n" + f"U={U}"
        return result


# matmul
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    x, y = promote_types_of_paddle_inputs(x, y)
    return ivy.matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)


# matrix_power
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def matrix_power(x, n, name=None):
    return ivy.matrix_power(x, n)


# mv
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def mv(x, vec, name=None):
    return ivy.dot(x, vec)


# norm
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def norm(x, p="fro", axis=None, keepdim=False, name=None):
    if axis is None and p is not None:
        if p == "fro":
            p = 2
        ret = ivy.vector_norm(x.flatten(), ord=p, axis=-1)
        if keepdim:
            ret = ret.reshape([1] * len(x.shape))
        return ret

    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    if isinstance(axis, int):
        if p == "fro":
            p = 2
        if p in [0, 1, 2, ivy.inf, -ivy.inf]:
            ret = ivy.vector_norm(x, ord=p, axis=axis, keepdims=keepdim)
        elif isinstance(p, (int, float)):
            ret = ivy.pow(
                ivy.sum(ivy.pow(ivy.abs(x), p), axis=axis, keepdims=keepdim),
                float(1.0 / p),
            )

    elif isinstance(axis, list) and len(axis) == 2:
        if p == 0:
            raise ValueError
        elif p == 1:
            ret = ivy.sum(ivy.abs(x), axis=axis, keepdims=keepdim)
        elif p in [2, "fro"]:
            ret = ivy.matrix_norm(x, ord="fro", axis=axis, keepdims=keepdim)
        elif p == ivy.inf:
            ret = ivy.max(ivy.abs(x), axis=axis, keepdims=keepdim)
        elif p == -ivy.inf:
            ret = ivy.min(ivy.abs(x), axis=axis, keepdims=keepdim)
        elif isinstance(p, (int, float)) and p > 0:
            ret = ivy.pow(
                ivy.sum(ivy.pow(ivy.abs(x), p), axis=axis, keepdims=keepdim),
                float(1.0 / p),
            )
        else:
            raise ValueError

    else:
        raise ValueError

    return ret


# pinv
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def pinv(x, rcond=1e-15, hermitian=False, name=None):
    # TODO: Add hermitian functionality
    return ivy.pinv(x, rtol=rcond)


# qr
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def qr(x, mode="reduced", name=None):
    return ivy.qr(x, mode=mode)


# solve
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def solve(x, y, name=None):
    return ivy.solve(x, y)


# transpose
@with_unsupported_dtypes({"2.6.0 and below": ("uint8", "int8", "int16")}, "paddle")
@to_ivy_arrays_and_back
def transpose(x, perm, name=None):
    return ivy.permute_dims(x, axes=perm)
