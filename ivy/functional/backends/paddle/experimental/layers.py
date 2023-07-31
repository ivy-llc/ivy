# global
from typing import Optional, Union, Tuple, List, Literal, Sequence
import paddle
from ivy.functional.ivy.layers import (
    _depth_max_pooling_helper,
    _handle_padding,
    _validate_max_pool_params,
)
from ivy.utils.assertions import check_kernel_padding_size
from ivy.utils.exceptions import IvyNotImplementedException, IvyValueError
from ivy.func_wrapper import (
    with_supported_device_and_dtypes,
    with_supported_dtypes,
)
from .. import backend_version

# local


def _determine_depth_max_pooling(x, kernel, strides, dims, data_format="channel_last"):
    # Determine depth pooling
    kernel, strides, depth_pooling = _depth_max_pooling_helper(
        x.shape, kernel, strides, dims=dims, data_format=data_format
    )
    if depth_pooling:
        x = paddle.transpose(x, (0, 2, 1, *range(3, dims + 2)))
    return x, kernel, strides, depth_pooling


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, backend_version)
def max_pool1d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dims = 1
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims=dims
    )
    if data_format == "NWC":
        x = paddle.transpose(x, perm=(0, 2, 1))
        kernel = [kernel[i] for i in [0, 2, 1]] if len(kernel) == (dims + 2) else kernel
        strides = (
            [strides[i] for i in [0, 2, 1]] if len(strides) == (dims + 2) else strides
        )
        padding = (
            [padding[i] for i in [0, 2, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    # Determine depthwise pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_first"
    )

    # TODO: Add support for pooling with dilation in the paddle backend.
    # It's currently not natively supported in the fromework.
    if max(dilation) > 1:
        raise NotImplementedError(
            "Max pooling with dilation is currently not supported in the 'paddle'"
            " backend"
        )

    padding = (
        [item for sublist in padding for item in sublist]
        if not isinstance(padding, str)
        else padding
    )  # to work directly with paddle's max_pool1d function
    res = paddle.nn.functional.max_pool1d(
        x, kernel, strides, padding=padding, ceil_mode=ceil_mode
    )

    if depth_pooling:
        res = paddle.transpose(res, perm=(0, 2, 1))
    if data_format == "NWC":
        res = paddle.transpose(res, perm=(0, 2, 1))
    return res


@with_supported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "float32",
                "float64",
            )
        }
    },
    backend_version,
)
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
    dtype = x.dtype

    x = x.astype("float32")
    if isinstance(strides, int):
        strides = (strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0])

    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    elif len(kernel) == 1:
        kernel = (kernel[0], kernel[0])

    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif len(dilation) == 1:
        dilation = (dilation[0], dilation[0])

    if isinstance(padding, int):
        padding = [(padding,) * 2] * 2
    elif isinstance(padding, tuple) and len(padding) == 1:
        padding = [(padding[0],) * 2] * 2
    elif isinstance(padding, tuple) and len(padding) == 2:
        padding = [(padding[0],) * 2, (padding[1],) * 2]

    if isinstance(padding, (tuple, list)):
        check_kernel_padding_size(kernel, padding)

    if data_format == "NHWC":
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
    x_shape = list(x.shape[2:])

    new_kernel = [kernel[i] + (kernel[i] - 1) * (dilation[i] - 1) for i in range(2)]

    if isinstance(padding, str):
        pad_h = _handle_padding(x_shape[0], strides[0], new_kernel[0], padding)
        pad_w = _handle_padding(x_shape[1], strides[1], new_kernel[1], padding)
        pad_list = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
    else:
        padding = (padding[1], padding[0])
        pad_list = [item for sublist in padding for item in sublist]

    x = paddle.nn.functional.pad(
        x,
        pad_list,
        value=float("-inf"),
    )

    res = paddle.nn.functional.max_pool2d(
        x, kernel_size=new_kernel, stride=strides, padding=0, ceil_mode=ceil_mode
    )

    if data_format == "NHWC":
        return paddle.transpose(res, perm=[0, 2, 3, 1]).astype(dtype)
    return res.astype(dtype)


def max_pool3d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dims = 3
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims=dims
    )

    if data_format == "NDHWC":
        x = paddle.transpose(x, perm=(0, 4, 1, 2, 3))
        kernel = (
            [kernel[i] for i in [0, 4, 1, 2, 3]]
            if len(kernel) == (dims + 2)
            else kernel
        )
        strides = (
            [strides[i] for i in [0, 4, 1, 2, 3]]
            if len(strides) == (dims + 2)
            else strides
        )
        padding = (
            [padding[i] for i in [0, 4, 1, 2, 3]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    # Determine depthwise pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_first"
    )

    # TODO: Add support for pooling with dilation in the paddle backend.
    # It's currently not natively supported in the fromework.
    if max(dilation) > 1:
        raise NotImplementedError(
            "Max pooling with dilation is currently not supported in the 'paddle'"
            " backend"
        )

    padding = (
        [item for sublist in padding for item in sublist]
        if not isinstance(padding, str)
        else padding
    )  # paddle's expected format
    res = paddle.nn.functional.max_pool3d(
        x, kernel, strides, padding=padding, ceil_mode=ceil_mode
    )

    if depth_pooling:
        res = paddle.transpose(res, perm=[0, 2, 1, 3, 4])
    if data_format == "NDHWC":
        res = paddle.transpose(res, perm=[0, 2, 3, 4, 1])
    return res


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
    if not isinstance(dim, int):
        raise IvyValueError(f"Expecting <class 'int'> instead of {type(dim)}")

    if n is None:
        n = x.shape[dim]

    if dim < -x.ndim or dim >= x.ndim:
        raise IvyValueError(
            f"Invalid dim {dim}, expecting a value ranging from {-x.ndim} to {x.ndim-1}"
        )

    if not isinstance(n, int):
        raise TypeError(f"Expecting int type for 'n', instead of {type(n)}")

    if n <= 1:
        raise IvyValueError(f"Invalid number of data points {n}, expecting more than 1")

    valid_norm_modes = ["backward", "ortho", "forward"]
    if norm not in valid_norm_modes:
        raise IvyValueError(
            f"Unrecognized normalization mode {norm}, expecting one of"
            f" {valid_norm_modes}"
        )

    if x.dtype in [paddle.int64, paddle.float64, paddle.complex128]:
        x = x.cast(paddle.complex128)
    else:
        x = x.cast(paddle.complex64)

    return paddle.fft.fft(x, n, dim, norm=norm)


@with_supported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "float32",
                "float64",
            )
        }
    },
    backend_version,
)
def dropout1d(
    x: paddle.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axis = data_format.index("C") - 3 + x.ndim
    return paddle.nn.functional.dropout(x, p=prob, axis=axis, training=training)


@with_supported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "float32",
                "float64",
            )
        }
    },
    backend_version,
)
def dropout2d(
    x: paddle.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axis = data_format.index("C") - 4 + x.ndim
    return paddle.nn.functional.dropout(x, p=prob, axis=axis, training=training)


@with_supported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "float32",
                "float64",
            )
        }
    },
    backend_version,
)
def dropout3d(
    x: paddle.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axis = data_format.index("C") - 5 + x.ndim
    return paddle.nn.functional.dropout(x, p=prob, axis=axis, training=training)


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
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: Optional[bool] = None,
    antialias: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def ifftn(
    x: paddle.Tensor,
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: Optional[str] = "backward",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.fft.ifftn(x, s, axes, norm)
