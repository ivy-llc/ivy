# global
from typing import Optional, Union, Tuple, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

# local
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "uint16",
                "bfloat16",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def median(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if input.ndim == 0:
        input = input.unsqueeze(0)
        return paddle.median(x=input, axis=axis).squeeze()
    elif input.ndim == 1:
        return paddle.median(x=input) if keepdims else paddle.median(x=input).squeeze()

    return paddle.median(x=input, axis=axis, keepdim=keepdims)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "float16", "complex64", "complex128")
        }
    },
    backend_version,
)
def nanmean(
    a: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if a.dtype not in [paddle.int64, paddle.float32, paddle.float64]:
        if dtype is None:
            dtype = a.dtype
        a = a.cast("float32")
        paddle.nanmean(x=a, axis=axis, keepdim=keepdims).cast(dtype)
    return paddle.nanmean(x=a, axis=axis, keepdim=keepdims).cast(dtype)


def quantile(
    a: paddle.Tensor,
    q: Union[paddle.Tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    raise IvyNotImplementedException()


def corrcoef(
    x: paddle.Tensor,
    /,
    *,
    y: Optional[paddle.Tensor] = None,
    rowvar: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nanmedian(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def unravel_index(
    indices: paddle.Tensor,
    shape: Tuple[int],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
