import math
from typing import Optional, Tuple
import numpy as np

import ivy


def diagflat(
    x: np.ndarray,
    /,
    *,
    offset: Optional[int] = 0,
    padding_value: Optional[float] = 0,
    align: Optional[str] = "RIGHT_LEFT",
    num_rows: Optional[int] = -1,
    num_cols: Optional[int] = -1,
    out: Optional[np.ndarray] = None,
):
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
    output_array += np.pad(
        diagonal_to_add,
        [
            (0, max([output_array.shape[0] - diagonal_to_add.shape[0], 0])),
            (0, max([output_array.shape[1] - diagonal_to_add.shape[1], 0])),
        ],
        mode="constant",
    )
    ret = output_array.astype(x.dtype)

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


def eig(x: np.ndarray, /) -> Tuple[np.ndarray, ...]:
    if ivy.dtype(x) == ivy.float16:
        x = x.astype(np.float32)
    e, v = np.linalg.eig(x)
    return e.astype(complex), v.astype(complex)
