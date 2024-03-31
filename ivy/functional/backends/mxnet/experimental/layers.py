# global
from typing import List, Optional, Union, Tuple, Literal, Sequence
import mxnet as mx

# local
from ivy.utils.exceptions import IvyNotImplementedException


def general_pool(
    inputs,
    init,
    reduce_fn,
    window_shape,
    strides,
    padding,
    dim,
    dilation=1,
    ceil_mode=False,
    count_include_pad=False,
):
    raise IvyNotImplementedException()


def max_pool1d(
    x: mx.nd.NDArray,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: Union[str, int, Tuple[int]],
    /,
    *,
    data_format: str = "NWC",
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def max_pool2d(
    x: mx.nd.NDArray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, Tuple[int], Tuple[int, int]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def max_pool3d(
    x: mx.nd.NDArray,
    kernel: Union[
        int, Tuple[int], Tuple[int, int, int], Tuple[int, int, int, int, int]
    ],
    strides: Union[
        int, Tuple[int], Tuple[int, int, int], Tuple[int, int, int, int, int]
    ],
    padding: Union[str, int, Tuple[int], Tuple[int, int, int]],
    /,
    *,
    data_format: str = "NDHWC",
    dilation: Union[int, Tuple[int], Tuple[int, int, int]] = 1,
    ceil_mode: bool = False,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def avg_pool1d(
    x: mx.nd.NDArray,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def avg_pool2d(
    x: mx.nd.NDArray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def avg_pool3d(
    x: mx.nd.NDArray,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def dct(
    x: mx.nd.NDArray,
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def fft(
    x: mx.nd.NDArray,
    dim: int,
    /,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def dropout1d(
    x: mx.nd.NDArray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def dropout2d(
    x: mx.nd.NDArray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def dropout3d(
    x: mx.nd.NDArray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def ifft(
    x: mx.nd.NDArray,
    dim: int,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def interpolate(
    x: mx.nd.NDArray,
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Literal[
        "linear",
        "bilinear",
        "trilinear",
        "nd",
        "nearest",
        "area",
        "nearest_exact",
        "tf_area",
        "tf_bicubic",
        "bicubic",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ] = "linear",
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: bool = False,
    antialias: bool = False,
    out: Optional[mx.nd.NDArray] = None,
):
    raise IvyNotImplementedException()


def rfft(
    x: mx.nd.NDArray,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()
