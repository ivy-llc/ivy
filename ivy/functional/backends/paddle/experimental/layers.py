# global
from typing import Optional, Union, Tuple, Literal, Sequence
import paddle
from ivy.functional.ivy.layers import _handle_padding
from ivy.utils.assertions import check_kernel_padding_size
from ivy.utils.exceptions import IvyNotImplementedException, IvyValueError
from ivy.func_wrapper import (
    with_supported_device_and_dtypes,
    with_unsupported_dtypes,
)
from .. import backend_version

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


@with_supported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
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
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0], strides[0])
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    elif len(kernel) == 1:
        kernel = (kernel[0], kernel[0], kernel[0])
    if data_format == "NDHWC":
        x = paddle.transpose(x, perm=[0, 2, 3, 4, 1])
    x_shape = list(x.shape[2:])
    pad_d = _handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_h = _handle_padding(x_shape[1], strides[1], kernel[1], padding)
    pad_w = _handle_padding(x_shape[2], strides[2], kernel[2], padding)
    x = paddle.nn.functional.pad(
        x,
        [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_d // 2,
            pad_d - pad_d // 2,
        ],
        value=float("-inf"),
        data_format="NDHWC",
    )
    if padding != "VALID" and padding != "SAME":
        raise ValueError(
            f'Invalid padding arg {padding}\nMust be one of: "VALID" or "SAME"'
        )
    res = paddle.nn.functional.max_pool3d(
        x, kernel_size=kernel, stride=strides, padding=0
    )
    if data_format == "NDHWC":
        res = paddle.transpose(res, perm=[0, 4, 1, 2, 3])
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
        "2.5.0 and below": {
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
        "2.5.0 and below": {
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


def stft(
    signal: Union[paddle.Tensor, int, Tuple[int]],
    n_fft: Union[int, Tuple[int]],
    frame_step: int,
    /,
    *,
    axis: Optional[int] = None,
    onesided:Optional[bool] = True,
    fs: Optional[float] = 1.0,
    window: Optional[Union[paddle.Tensor, list, str, Tuple[int]]] = None,
    win_length: Optional[int] = None,
    noverlap: Optional[int] = None,
    center: Optional[bool] = True,
    pad_mode: Optional[str] = "reflect",
    normalized: Optional[bool] = False,
    detrend: Optional[Union[str, callable, bool]] = False,
    return_complex: Optional[bool] = True,
    boundary: Optional[str] = 'zeros',
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.signal.stft(
         signal,
         n_fft,
         frame_step,
         frame_length,
         window,
         center,
         pad_mode,
         normalized,
         onesided,
    )

    return paddle.fft.ifftn(x, s, axes, norm)


@with_unsupported_dtypes(
    {"2.5.0 and below": ("bfloat16", "float16", "complex64", "complex128", "bool")},
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
