# global
from typing import Optional, Union, Tuple, List, Literal, Sequence
import paddle
from ivy.functional.ivy.layers import (
    _depth_max_pooling_helper,
    _validate_max_pool_params,
)
from ivy.utils.exceptions import IvyNotImplementedException, IvyValueError
from ivy.func_wrapper import (
    with_supported_device_and_dtypes,
    with_unsupported_dtypes,
    with_supported_dtypes,
)
from .. import backend_version

# local


def _determine_depth_max_pooling(x, kernel, strides, dims, data_format="channel_first"):
    # Determine depth pooling
    kernel, strides, depth_pooling = _depth_max_pooling_helper(
        x.shape, kernel, strides, dims=dims, data_format=data_format
    )
    if depth_pooling:
        x = paddle.transpose(x, (0, 2, 1, *range(3, dims + 2)))
    return x, kernel, strides, depth_pooling


@with_supported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    backend_version,
)
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
        "2.5.1 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    backend_version,
)
def max_pool2d(
    x: paddle.Tensor,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dims = 2
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims=dims
    )

    if data_format == "NHWC":
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        kernel = (
            [kernel[i] for i in [0, 3, 1, 2]] if len(kernel) == (dims + 2) else kernel
        )
        strides = (
            [strides[i] for i in [0, 3, 1, 2]]
            if len(strides) == (dims + 2)
            else strides
        )
        padding = (
            [padding[i] for i in [0, 3, 1, 2]]
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
    res = paddle.nn.functional.max_pool2d(
        x, kernel, strides, padding=padding, ceil_mode=ceil_mode
    )

    if depth_pooling:
        res = paddle.transpose(res, perm=[0, 2, 1, 3])
    if data_format == "NHWC":
        res = paddle.transpose(res, perm=[0, 2, 3, 1])
    return res


@with_supported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    backend_version,
)
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
        "2.5.1 and below": {
            "cpu": ("bfloat16", "float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
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
        "2.5.1 and below": {
            "cpu": ("bfloat16", "float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
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
        "2.5.1 and below": {
            "cpu": ("bfloat16", "float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
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


def adaptive_max_pool2d(
    input: paddle.Tensor, output_size: Union[Sequence[int], int]
) -> paddle.Tensor:
    squeeze = input.ndim == 3
    x = paddle.unsqueeze(input, axis=0) if squeeze else input
    ret = paddle.nn.functional.adaptive_max_pool2d(x, output_size)
    return paddle.squeeze(ret, axis=0) if squeeze else ret


def ifftn(
    x: paddle.Tensor,
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: Optional[str] = "backward",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.fft.ifftn(x, s, axes, norm)


@with_unsupported_dtypes(
    {"2.5.1 and below": ("bfloat16", "float16", "complex64", "complex128", "bool")},
    backend_version,
)
def rfftn(
    x: paddle.Tensor,
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: Optional[str] = "backward",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    result = paddle.fft.rfftn(x, s, axes, norm)
    return result.astype("complex128")


@with_supported_dtypes(
    {
        "2.5.1 and below": (
            "complex64",
            "complex128",
        )
    },
    backend_version,
)
def fft2(
    x: paddle.Tensor,
    *,
    dim: Optional[Union[int, Tuple[int]]] = None,
    norm: Optional[str] = "backward",
    s: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    res = paddle.fft.fft2(x, s, dim, norm)
    return res.astype("complex128")


def deform_conv2d(
    x: paddle.Tensor,
    offset: paddle.Tensor,
    weight: paddle.Tensor,
    *,
    bias: Optional[paddle.Tensor] = None,
    stride: Union[int, Tuple[int]] = (1, 1),
    padding: Union[int, Tuple[int]] = (0, 0),
    dilation: Union[int, Tuple[int]] = (1, 1),
    mask: Optional[paddle.Tensor] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if offset.shape[1] % (2 * weight.shape[2] * weight.shape[3]) != 0:
        raise Exception("offset_groups must be integer")
    offset_groups = int(offset.shape[1] // (2 * weight.shape[2] * weight.shape[3]))
    groups = int(x.shape[1] // weight.shape[1])
    conv_out = paddle.vision.ops.deform_conv2d(
        x=x,
        offset=offset,
        weight=weight,
        bias=bias,
        mask=mask,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        deformable_groups=offset_groups,
    )
    return conv_out
