from typing import Union, Optional, Tuple, List, NamedTuple
import tensorflow as tf

import ivy

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size

from ivy.func_wrapper import with_supported_dtypes
from .. import backend_version


@with_supported_dtypes({"2.9.1 and below": ("float32", "float64")}, backend_version)
def eigh_tridiagonal(
    alpha: Union[tf.Tensor, tf.Variable],
    beta: Union[tf.Tensor, tf.Variable],
    /,
    *,
    eigvals_only: bool = True,
    select: str = 'a',
    select_range: Optional[Union[Tuple[int], List[int], tf.Tensor, tf.Variable]] = None,
    tol: Optional[float] = None,
) -> Union[Union[tf.Tensor, tf.Variable], Tuple[Union[tf.Tensor, tf.Variable]]]:
    if eigvals_only:
        return tf.linalg.eigh_tridiagonal(
            alpha,
            beta,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            tol=tol
        )
        
    result_tuple = NamedTuple(
        "eigh",
        [
            ("eigenvalues", Union[tf.Tensor, tf.Variable]),
            ("eigenvectors", Union[tf.Tensor, tf.Variable]),
        ],
    )
    eigenvalues, eigenvectors = tf.linalg.eigh_tridiagonal(
        alpha,
        beta,
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol
    )

    return result_tuple(eigenvalues, eigenvectors)


def diagflat(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    offset: Optional[int] = 0,
    padding_value: Optional[float] = 0,
    align: Optional[str] = "RIGHT_LEFT",
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
    return tf.math.exp(x)


def eig(
    x: Union[tf.Tensor],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor]:
    if not ivy.dtype(x) in (ivy.float32, ivy.float64, ivy.complex64, ivy.complex128):
        return tf.linalg.eig(tf.cast(x, tf.float64))
    return tf.linalg.eig(x)


def eigvals(
    x: Union[tf.Tensor],
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
