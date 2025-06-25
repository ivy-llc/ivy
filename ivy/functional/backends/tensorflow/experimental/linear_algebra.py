from typing import Union, Optional, Tuple, List, Sequence
import tensorflow as tf
from functools import reduce as _reduce
from collections import namedtuple
import ivy

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size

from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from .. import backend_version


@with_unsupported_dtypes(
    {"2.15.0 and below": ("int", "float16", "bfloat16")}, backend_version
)
def eigh_tridiagonal(
    alpha: Union[tf.Tensor, tf.Variable],
    beta: Union[tf.Tensor, tf.Variable],
    /,
    *,
    eigvals_only: bool = True,
    select: str = "a",
    select_range: Optional[
        Union[Tuple[int, int], List[int], tf.Tensor, tf.Variable]
    ] = None,
    tol: Optional[float] = None,
) -> Union[
    tf.Tensor,
    tf.Variable,
    Tuple[Union[tf.Tensor, tf.Variable], Union[tf.Tensor, tf.Variable]],
]:
    return tf.linalg.eigh_tridiagonal(
        alpha,
        beta,
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol,
    )


def diagflat(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    offset: int = 0,
    padding_value: float = 0,
    align: str = "RIGHT_LEFT",
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if len(x.shape) > 1:
        x = tf.reshape(x, [-1])

    if num_rows is None:
        num_rows = -1
    if num_cols is None:
        num_cols = -1

    ret = tf.linalg.diag(
        x,
        name="diagflat",
        k=offset,
        num_rows=num_rows,
        num_cols=num_cols,
        padding_value=padding_value,
        align=align,
    )

    if ivy.exists(out):
        ivy.inplace_update(out, ret)

    return ret


def kron(
    a: Union[tf.Tensor, tf.Variable],
    b: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    a_ndims = a.shape.ndims
    b_ndims = b.shape.ndims

    if a_ndims <= 2 and b_ndims <= 2:
        return tf.experimental.numpy.kron(a, b)

    a_shape = tf.shape(a)
    b_shape = tf.shape(b)

    if a_ndims is None or b_ndims is None:
        raise ValueError("Cannot determine number of dimensions for input arrays.")

    if a_ndims == 0 or b_ndims == 0:
        return a * b

    if a_ndims < b_ndims:
        a = tf.reshape(
            a, tf.concat([tf.ones(b_ndims - a_ndims, dtype=tf.int32), a_shape], 0)
        )
        a_shape = tf.shape(a)
        a_ndims = b_ndims
    elif b_ndims < a_ndims:
        b = tf.reshape(
            b, tf.concat([tf.ones(a_ndims - b_ndims, dtype=tf.int32), b_shape], 0)
        )
        b_shape = tf.shape(b)

    ndims = a_ndims
    if ndims == 0:
        return a * b

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    a_axes = alphabet[:ndims]
    b_axes = alphabet[ndims : 2 * ndims]
    einsum_str = f"{a_axes},{b_axes}->" + "".join(
        [sub[item] for sub in zip(a_axes, b_axes) for item in range(2)]
    )
    promoted_dtype = tf.experimental.numpy.result_type(a.dtype, b.dtype)
    a = tf.cast(a, promoted_dtype)
    b = tf.cast(b, promoted_dtype)
    res = tf.einsum(einsum_str, a, b)

    final_shape = [a_shape[i] * b_shape[i] for i in range(ndims)]
    return tf.reshape(res, final_shape)


def matrix_exp(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.linalg.expm(x)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "complex",
            "float32",
            "float64",
        )
    },
    backend_version,
)
def eig(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor]:
    return tf.linalg.eig(x)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "complex",
            "float32",
            "float64",
        )
    },
    backend_version,
)
def eigvals(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.linalg.eigvals(x)


def adjoint(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    _check_valid_dimension_size(x)
    return tf.linalg.adjoint(x)


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int", "float16", "bfloat16", "float64")}, backend_version
)
def solve_triangular(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    upper: bool = True,
    adjoint: bool = False,
    unit_diagonal: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # Multiplying with a mask matrix can stop gradients on the diagonal.
    if unit_diagonal:
        w = tf.constant(tf.eye(x1.shape[-2], batch_shape=x1.shape[:-2], dtype=x1.dtype))
        x1 = w + (1 - w) * x1
    return tf.linalg.triangular_solve(x1, x2, lower=not upper, adjoint=adjoint)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        )
    },
    backend_version,
)
def multi_dot(
    x: Sequence[Union[tf.Tensor, tf.Variable]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> tf.Tensor:
    # This implementation simply chains tf.tensordot multiple times
    # TODO: reimplement this function once tf adds multi_dot or inplace updates
    if len(x) < 2:
        raise ValueError("Expecting at least two tensors.")
    dot_out = _reduce(tf.matmul, x)
    return dot_out


@with_unsupported_dtypes({"2.15.0 and below": ("float16", "bfloat16")}, backend_version)
def cond(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    p: Optional[Union[None, int, str]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    svd = tf.linalg.svd(x, compute_uv=False)
    if len(x.shape) >= 3:
        ax = len(x.shape) // 2
    elif len(x.shape) >= 3 and p == -1:
        ax = [-1, -2]
    else:
        ax = None
    if p is None or p == 2:
        k = tf.reduce_max(svd, axis=ax) / tf.reduce_min(svd, axis=ax)
    elif p == "nuc":
        svd_inv = tf.linalg.svd(tf.linalg.inv(x), compute_uv=False)
        k = tf.reduce_sum(svd, axis=ax) * tf.reduce_sum(svd_inv, axis=ax)
    elif p == "fro":
        k = tf.norm(x, ord="euclidean", axis=[-2, -1]) * tf.norm(
            tf.linalg.inv(x), ord="euclidean", axis=[-2, -1]
        )
    elif p < 0:
        if p == -1:
            k = tf.reduce_min(
                tf.reduce_sum(tf.abs(x), axis=0), axis=ax
            ) * tf.reduce_min(tf.reduce_sum(tf.abs(tf.linalg.inv(x)), axis=0), axis=ax)
        elif p == -2:
            k = tf.reduce_min(svd, axis=ax) / tf.reduce_max(svd, axis=ax)
        elif p == -float("inf"):
            k = tf.reduce_min(
                tf.reduce_sum(tf.abs(x), axis=1), axis=ax
            ) * tf.reduce_min(tf.reduce_sum(tf.abs(tf.linalg.inv(x)), axis=1), axis=ax)
    else:
        k = tf.norm(x, ord=p, axis=[-2, -1]) * tf.norm(
            tf.linalg.inv(x), ord=p, axis=[-2, -1]
        )
    return k


@with_unsupported_dtypes(
    {"2.15.0 and below": ("integer", "float16", "bfloat16")}, backend_version
)
def lu_factor(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    pivot: Optional[bool] = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    ret = tf.linalg.lu(x)
    ret_tuple = namedtuple("lu_factor", ["LU", "p"])
    return ret_tuple(ret.lu, ret.p)


def lu_solve(
    lu: Union[tf.Tensor, tf.Variable],
    p: Union[tf.Tensor, tf.Variable],
    b: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    p = p - 1

    # convert lapack pivots to permutation vector
    n = tf.shape(lu)[-1]
    perm_vector = tf.range(n, dtype=p.dtype)

    def body(i, perm):
        p_i = p[i]
        val_at_i = perm[i]
        val_at_p_i = perm[p_i]
        indices = tf.expand_dims(tf.stack([i, p_i]), axis=-1)
        updates = tf.stack([val_at_p_i, val_at_i])
        perm = tf.tensor_scatter_nd_update(perm, indices, updates)
        return i + 1, perm

    num_pivots = tf.shape(p)[-1]
    _, permutation_vector = tf.while_loop(
        lambda i, perm: i < num_pivots,
        body,
        [tf.constant(0, dtype=p.dtype), perm_vector],
        shape_invariants=[tf.TensorShape([]), perm_vector.get_shape()],
    )
    return tf.linalg.lu_solve(lu, permutation_vector, b)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "complex64",
            "complex128",
            "bfloat16",
        )
    },
    backend_version,
)
def dot(
    a: tf.Tensor,
    b: tf.Tensor,
    /,
    *,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    a, b = ivy.promote_types_of_inputs(a, b)
    return tf.experimental.numpy.dot(a, b)
