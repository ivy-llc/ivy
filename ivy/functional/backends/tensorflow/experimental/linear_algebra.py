from typing import Union, Optional, Tuple, List, Sequence
import tensorflow as tf
from functools import reduce as _reduce

import ivy

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size

from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.utils.exceptions import IvyNotImplementedException
from .. import backend_version


@with_unsupported_dtypes(
    {"2.13.0 and below": ("int", "float16", "bfloat16")}, backend_version
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
    return tf.experimental.numpy.kron(a, b)


def matrix_exp(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.linalg.expm(x)


def eig(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor]:
    if not ivy.dtype(x) in (ivy.float32, ivy.float64, ivy.complex64, ivy.complex128):
        return tf.linalg.eig(tf.cast(x, tf.float64))
    return tf.linalg.eig(x)


def eigvals(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Union[tf.Tensor, tf.Variable]:
    if not ivy.dtype(x) in (ivy.float32, ivy.float64, ivy.complex64, ivy.complex128):
        return tf.linalg.eigvals(tf.cast(x, tf.float64))
    return tf.linalg.eigvals(x)


def adjoint(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    _check_valid_dimension_size(x)
    return tf.linalg.adjoint(x)


@with_supported_dtypes(
    {
        "2.13.0 and below": (
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


@with_unsupported_dtypes({"1.25.0 and below": ("float16", "bfloat16")}, backend_version)
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


def lu_factor(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    pivot: Optional[bool] = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor]:
    raise IvyNotImplementedException()


def dot(
    a: tf.Tensor,
    b: tf.Tensor,
    /,
    *,
    out: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    return tf.tensordot(a, b, out=out)


dot.support_native_out = True
