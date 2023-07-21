import math
from typing import Optional, Tuple, Sequence, Union
import jax.numpy as jnp
import jax.scipy.linalg as jla
from ivy.functional.backends.jax import JaxArray
import ivy

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size
from ivy.utils.exceptions import IvyNotImplementedException


def diagflat(
    x: JaxArray,
    /,
    *,
    offset: int = 0,
    padding_value: float = 0,
    align: str = "RIGHT_LEFT",
    num_rows: int = -1,
    num_cols: int = -1,
    out: Optional[JaxArray] = None,
):
    if len(x.shape) > 1:
        x = jnp.ravel(x)

    # Trying to avoid segfaults
    x = jnp.copy(x)
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

    output_array = jnp.full(output_shape, padding_value)

    diag_len = max(min(num_rows, num_cols) - abs(offset) + 1, 1)

    if len(x) < diag_len:
        x = jnp.array(list(x) + [padding_value] * max((diag_len - len(x), 0)))

    temp = x - jnp.full(x.shape, padding_value)
    diagonal_to_add = jnp.diag(temp, k=offset)

    diagonal_to_add = diagonal_to_add[tuple(slice(0, n) for n in output_array.shape)]
    output_array += jnp.pad(
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
    a: JaxArray,
    b: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.kron(a, b)


def matrix_exp(
    x: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jla.expm(x)


def eig(
    x: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> Tuple[JaxArray]:
    return jnp.linalg.eig(x)


def eigvals(x: JaxArray, /) -> JaxArray:
    if not ivy.dtype(x) in (ivy.float32, ivy.float64, ivy.complex64, ivy.complex128):
        x = x.astype(jnp.float64)
    return jnp.linalg.eigvals(x)


def adjoint(
    x: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    _check_valid_dimension_size(x)
    axes = list(range(len(x.shape)))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return jnp.conjugate(jnp.transpose(x, axes=axes))


def multi_dot(
    x: Sequence[JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.linalg.multi_dot(x)


def dot(x1: JaxArray, x2: JaxArray, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.dot(a=x1, b=x2)


dot.support_native_out = True


def cond(
    x: JaxArray,
    /,
    *,
    p: Optional[Union[int, str, None]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.linalg.cond(x, p=p)


def lu_factor(
    x: JaxArray,
    /,
    *,
    pivot: Optional[bool] = True,
    out: Optional[JaxArray] = None,
) -> Tuple[JaxArray]:
    raise IvyNotImplementedException()
