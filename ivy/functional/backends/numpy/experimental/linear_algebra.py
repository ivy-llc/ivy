import math
from typing import Optional, Tuple, Sequence, Union, Any
import numpy as np

import ivy
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from ivy.utils.exceptions import IvyNotImplementedException
from .. import backend_version

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size


def diagflat(
    x: np.ndarray,
    /,
    *,
    offset: int = 0,
    padding_value: float = 0,
    align: str = "RIGHT_LEFT",
    num_rows: int = -1,
    num_cols: int = -1,
    out: Optional[np.ndarray] = None,
):
    out_dtype = x.dtype if out is None else out.dtype
    if len(x.shape) > 1:
        x = np.ravel(x)

    if math.prod(x.shape) == 1 and offset == 0 and num_rows <= 1 and num_cols <= 1:
        return x

    # This is used as part of Tensorflow's shape calculation
    # See their source code to see what they're doing with it
    lower_diag_index = offset
    upper_diag_index = lower_diag_index

    x_shape = x.shape
    x_rank = len(x_shape)

    num_diags = upper_diag_index - lower_diag_index + 1
    max_diag_len = x_shape[x_rank - 1]

    min_num_rows = max_diag_len - min(upper_diag_index, 0)
    min_num_cols = max_diag_len + max(lower_diag_index, 0)

    if num_rows == -1 and num_cols == -1:
        num_rows = max(min_num_rows, min_num_cols)
        num_cols = num_rows
    elif num_rows == -1:
        num_rows = min_num_rows
    elif num_cols == -1:
        num_cols = min_num_cols

    output_shape = list(x_shape)
    if num_diags == 1:
        output_shape[x_rank - 1] = num_rows
        output_shape.append(num_cols)
    else:
        output_shape[x_rank - 2] = num_rows
        output_shape[x_rank - 1] = num_cols

    output_array = np.full(output_shape, padding_value)

    diag_len = max(min(num_rows, num_cols) - abs(offset) + 1, 1)

    if len(x) < diag_len:
        x = np.array(list(x) + [padding_value] * max((diag_len - len(x), 0)))

    diagonal_to_add = np.diag(x - np.full_like(x, padding_value), k=offset)

    diagonal_to_add = diagonal_to_add[tuple(slice(0, n) for n in output_array.shape)]
    ret = diagonal_to_add.astype(out_dtype)

    if ivy.exists(out):
        ivy.inplace_update(out, ret)

    return ret


def kron(
    a: np.ndarray,
    b: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.kron(a, b)


kron.support_native_out = False


@with_supported_dtypes(
    {"1.26.3 and below": ("float32", "float64", "complex64", "complex128")},
    backend_version,
)
def matrix_exp(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    eig_vals, eig_vecs = np.linalg.eig(x)
    exp_diag = np.exp(eig_vals)
    exp_diag_mat = np.diag(exp_diag)
    exp_mat = eig_vecs @ exp_diag_mat @ np.linalg.inv(eig_vecs)
    return exp_mat.astype(x.dtype)


@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, backend_version)
def eig(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    e, v = np.linalg.eig(x)
    return e.astype(complex), v.astype(complex)


eig.support_native_out = False


@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, backend_version)
def eigvals(x: np.ndarray, /) -> np.ndarray:
    e = np.linalg.eigvals(x)
    return e.astype(complex)


eigvals.support_native_out = False


def adjoint(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    _check_valid_dimension_size(x)
    axes = list(range(len(x.shape)))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return np.conjugate(np.transpose(x, axes=axes))


_adjoint = adjoint


def solve_triangular(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    upper: bool = True,
    adjoint: bool = False,
    unit_diagonal: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    # NumPy does not expose an API for `trsm`, so we have to implement substitution
    # in Python. There is no need to support gradients for this backend.
    # Pre: `x1` is square, `x1` and `x2` have the same number `n` of rows.
    n = x1.shape[-2]
    ret = x2.copy()

    if adjoint:
        x1 = _adjoint(x1)
        upper = not upper

    if unit_diagonal:
        for i in range(n):
            x1[..., i, i] = 1

    if upper:
        for i in reversed(range(n)):
            ret[..., i, :] /= x1[..., i, np.newaxis, i]
            ret[..., :i, :] -= x1[..., :i, np.newaxis, i] * ret[..., np.newaxis, i, :]
    else:
        for i in range(n):
            ret[..., i, :] /= x1[..., i, np.newaxis, i]
            ret[..., i + 1 :, :] -= (
                x1[..., i + 1 :, np.newaxis, i] * ret[..., np.newaxis, i, :]
            )

    return ret


def multi_dot(
    x: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.array] = None,
) -> np.ndarray:
    return np.linalg.multi_dot(x, out=out)


multi_dot.support_native_out = True


def cond(
    x: np.ndarray,
    /,
    *,
    p: Optional[Union[None, int, str]] = None,
    out: Optional[np.ndarray] = None,
) -> Any:
    return np.linalg.cond(x, p=p)


cond.support_native_out = False


def lu_factor(
    x: np.ndarray,
    /,
    *,
    pivot: Optional[bool] = True,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray]:
    raise IvyNotImplementedException()


def lu_solve(
    lu: Tuple[np.ndarray],
    p: np.ndarray,
    b: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    raise IvyNotImplementedException()


def dot(
    a: np.ndarray,
    b: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.dot(a, b, out=out)


dot.support_native_out = True
