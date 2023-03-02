"""Collection of Paddle activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union

# global
import paddle
import paddle.nn.functional as F

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def relu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:

    return F.relu(x)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def leaky_relu(
    x: paddle.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    return F.leaky_relu(x, negative_slope=alpha)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def gelu(
    x: paddle.Tensor,
    /,
    *,
    approximate: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    return F.gelu(x, approximate=approximate)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def sigmoid(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:

    return F.sigmoid(x)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if axis is None:
        axis = -1
    return F.softmax(x, axis=axis)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def softplus(
    x: paddle.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    threshold = threshold if threshold is not None else 20
    beta = beta if beta is not None else 1

    return F.softplus(x, beta=beta, threshold=threshold)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def log_softmax(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
):
    if axis is None:
        axis = -1
    return F.log_softmax(x, axis=axis)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def mish(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:

    return F.mish(x)
