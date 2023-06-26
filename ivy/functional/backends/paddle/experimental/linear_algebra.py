# global
import paddle
from typing import Optional, Tuple, Union, Any

# local
from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from ivy.utils.exceptions import IvyNotImplementedException
from .. import backend_version


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("int8", "int16", "uint8", "float16")}}, backend_version
)
def diagflat(
    x: paddle.Tensor,
    /,
    *,
    offset: Optional[int] = 0,
    padding_value: Optional[float] = 0,
    align: Optional[str] = "RIGHT_LEFT",
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
):
    diag = paddle.diag(x.flatten(), padding_value=padding_value, offset=offset)
    num_rows = num_rows if num_rows is not None else diag.shape[0]
    num_cols = num_cols if num_cols is not None else diag.shape[1]

    if num_rows < diag.shape[0]:
        diag = diag[:num_rows, :]
    if num_cols < diag.shape[1]:
        diag = diag[:, :num_cols]

    if diag.shape == [num_rows, num_cols]:
        return diag
    else:
        return paddle.nn.Pad2D(
            padding=(0, num_rows - diag.shape[0], 0, num_cols - diag.shape[1]),
            mode="constant",
            value=padding_value,
        )(diag)


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("int8", "uint8", "int16")}}, backend_version
)
def kron(
    a: paddle.Tensor,
    b: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.kron(a, b)


def matrix_exp(
    x: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # TODO: this is elementwise exp, should be changed to matrix exp ASAP
    return paddle.exp(x)


def eig(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor]:
    return paddle.linalg.eig(x)


def eigvals(x: paddle.Tensor, /) -> paddle.Tensor:
    return paddle.linalg.eig(x)[0]


def adjoint(
    x: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    _check_valid_dimension_size(x)
    return paddle.moveaxis(x, -2, -1).conj()


def cond(
    x: paddle.Tensor,
    /,
    *,
    p: Optional[Union[None, int, str]] = None,
    out: Optional[paddle.Tensor] = None,
) -> Any:
    raise IvyNotImplementedException()


def cov(
    x1: paddle.Tensor,
    x2: paddle.Tensor = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[paddle.Tensor] = None,
    aweights: Optional[paddle.Tensor] = None,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    if fweights is not None:
        fweights = fweights.astype("float64")

    if aweights is not None:
        aweights = aweights.astype("float64")

    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be an integer")

    if len(x1.shape) > 2:
        raise ValueError("x1 has more than 2 dimensions")

    if x2 is not None:
        if len(x2.shape) > 2:
            raise ValueError("x2 has more than 2 dimensions")

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    if dtype is None:
        x1 = x1.astype("float64")
        if x2 is not None:
            x2 = x2.astype("float64")
    else:
        x1 = x1.astype(dtype)
        if x2 is not None:
            x2 = x2.astype(dtype)

    X = x1
    if not rowVar and X.shape[0] != 1:
        X = paddle.transpose(X, perm=tuple(range(len(X.shape) - 1, -1, -1)))

    if x2 is not None:
        if not rowVar and x2.shape[0] != 1:
            x2 = paddle.transpose(x2, perm=tuple(range(len(x2.shape) - 1, -1, -1)))
        if len(x2.shape) > 1:
            X = paddle.concat([X, x2], axis=0)
        else:
            X = paddle.stack([X, x2], axis=0)

    if not rowVar:
        X = paddle.transpose(X, perm=tuple(range(len(X.shape) - 1, -1, -1)))

    return paddle.linalg.cov(
        X, rowvar=rowVar, ddof=ddof, fweights=fweights, aweights=aweights
    )
