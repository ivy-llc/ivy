# global
from typing import Optional, Union, Tuple, Literal, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.functional.ivy.layers import _handle_padding

# local


def max_pool1d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = x.dtype
    x = x.astype("float64")
    if isinstance(strides, int):
        strides = (strides,)
    elif len(strides) == 1:
        strides = (strides[0],)

    if isinstance(kernel, int):
        kernel = (kernel,)
    elif len(kernel) == 1:
        kernel = (kernel[0],)

    if data_format == "NWC":
        x = paddle.transpose(x, perm=(0, 2, 1))
    x_shape = x.shape[2]
    pad_w = _handle_padding(x_shape, strides[0], kernel[0], padding)
    x = paddle.nn.functional.pad(
        x, pad=[pad_w // 2, pad_w - pad_w // 2], value=float("-inf"), data_format="NCL"
    )

    res = paddle.nn.functional.max_pool1d(x, kernel, strides, padding="valid")

    if data_format == "NWC":
        res = paddle.transpose(res, perm=(0, 2, 1))
    return res.astype(dtype)


def max_pool2d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, Tuple[int], Tuple[int, int]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def max_pool3d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def avg_pool1d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def avg_pool2d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def avg_pool3d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def dct(
    x: paddle.Tensor,
    /,
    *,
    type: Optional[Literal[1, 2, 3, 4]] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def fft(
    x: paddle.Tensor,
    dim: int,
    /,
    *,
    norm: Optional[str] = "backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def dropout1d(
    x: paddle.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def ifft(
    x: paddle.Tensor,
    dim: int,
    *,
    norm: Optional[str] = "backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def embedding(
    weights: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    max_norm: Optional[int] = None,
    out=None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def interpolate(
    x: paddle.Tensor,
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Optional[Literal["linear", "bilinear", "trilinear"]] = "linear",
    align_corners: Optional[bool] = None,
    antialias: Optional[bool] = False,
):
    raise IvyNotImplementedException()
