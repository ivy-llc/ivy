# global
from typing import Optional, Union, Tuple, Literal, Sequence
import mxnet as mx

# local
from ivy.func_wrapper import handle_mixed_function
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
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
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
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def avg_pool1d(
    x: mx.nd.NDArray,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    raise IvyNotImplementedException()


def avg_pool2d(
    x: mx.nd.NDArray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
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
    padding: str,
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


@handle_mixed_function(
    lambda *args, mode="linear", scale_factor=None, recompute_scale_factor=None, align_corners=None, **kwargs: (  # noqa: E501
        not align_corners
        and mode
        not in [
            "area",
            "nearest",
            "tf_area",
            "mitchellcubic",
            "gaussian",
            "bicubic",
        ]
        and recompute_scale_factor
    )
)
def interpolate(
    x: mx.nd.NDArray,
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Literal[
        "linear",
        "bilinear",
        "trilinear",
        "nearest",
        "area",
        "nearest_exact",
        "tf_area",
        "bicubic_tensorflow" "bicubic",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ] = "linear",
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: Optional[bool] = None,
    antialias: bool = False,
    out: Optional[mx.nd.NDArray] = None,
):
    raise IvyNotImplementedException()
