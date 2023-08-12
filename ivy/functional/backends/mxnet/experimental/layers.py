# global
from typing import Optional, Union, Tuple, Literal, Sequence
import mxnet as mx

# local
from ivy.func_wrapper import handle_partial_mixed_function
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


@handle_partial_mixed_function(
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


def deform_conv2d(
    x: mx.nd.NDArray,
    offset: mx.nd.NDArray,
    weight: mx.nd.NDArray,
    *,
    bias: Optional[mx.nd.NDArray] = None,
    stride: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, int]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    mask: Optional[mx.nd.NDArray] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if offset.shape[1] % (2 * weight.shape[2] * weight.shape[3]) != 0:
        raise Exception("offset_groups must be integer")
    offset_groups = int(offset.shape[1] // (2 * weight.shape[2] * weight.shape[3]))
    groups = int(x.shape[1] // weight.shape[1])

    if mask is None:
        return mx.nd.contrib.DeformableConvolution(
            x,
            offset,
            weight,
            bias,
            kernel=weight.shape[2:],
            stride=stride,
            pad=padding,
            dilate=dilation,
            num_filter=weight.shape[0],
            num_group=groups,
            num_deformable_group=offset_groups,
        )
    else:
        return mx.nd.contrib.ModulatedDeformableConvolution(
            x,
            offset,
            mask,
            weight,
            bias,
            kernel=weight.shape[2:],
            stride=stride,
            pad=padding,
            dilate=dilation,
            num_filter=weight.shape[0],
        )
