# global
from typing import Optional, Union, Tuple, Literal, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.functional.ivy.layers import _handle_padding
from ivy.utils.assertions import check_kernel_padding_size
from ivy.func_wrapper import (
    with_supported_dtypes,
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


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "float32",
            "float64",
        )
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


def dropout2d(
    x: paddle.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def dropout3d(
    x: paddle.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
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


def ifftn(
    x: paddle.Tensor,
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: Optional[str] = "backward",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.fft.ifftn(x, s, axes, norm)


def overlap_and_add(
    signal: paddle.Tensor,
    frame_step: int,
    name: str = None,
) -> paddle.Tensor:
    # convert to tensor for signal
    signal = paddle.to_tensor(signal)

    # shape limit check
    if len(signal.shape) < 2:
        raise ValueError("input must be at least 2-D")

    # check dtype for frame_step as integer
    if type(frame_step) not in (int,):
        raise ValueError("frame_step must be an integer")

    # Paddle Error
    # RuntimeError: (NotFound) The kernel with key (CPU, NCHW, float16) of kernel
    # `pad` is not registered.
    signal_dtype = None
    if signal.dtype == paddle.float16:
        signal_dtype = paddle.float16
        signal = signal.astype(paddle.float32)

    # get signal shape as constant value
    signal_shape = list(signal.shape)

    # get outer_dimensions [:-2]
    outer_dimensions = signal_shape[:-2]

    # get outer_rank [:-2]
    outer_rank = len(outer_dimensions)

    # make func for full_shape
    def full_shape(inner_shape):
        return outer_dimensions + inner_shape

    frame_length = signal_shape[-1]
    frames = signal_shape[-2]

    # Compute output length
    output_size = frame_length + (frames - 1) * frame_step

    # If frame_length is equal to frame_step, there's no overlap so just
    # reshape the tensor.
    if frame_step and signal.shape is not None and frame_step == signal.shape[-1]:
        output_shape = full_shape([output_size])
        return paddle.reshape(signal, output_shape)

    # The following code is documented using this example:
    #
    # frame_step = 2
    # signal.shape = (3, 5)
    # a b c d e
    # f g h i j
    # k l m n o

    # Compute the number of segments per frame.
    segments = -(-frame_length // frame_step)  # Divide and round up.

    # Pad the frame_length dimension to a multiple of the frame step.
    # Pad the frames dimension by `segments` so that signal.shape = (6, 6)
    # a b c d e 0
    # f g h i j 0
    # k l m n o 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    paddings = [
        0,
        segments,
        0,
        segments * frame_step - frame_length,
    ]  # zero padding for frames
    outer_paddings = [0] * outer_rank * 2  # dummy for outer_rank [0, 0]
    outer_paddings += paddings

    signal = paddle.nn.functional.pad(signal, paddings, mode="constant", value=0)

    # Reshape so that signal.shape = (3, 6, 2)
    # ab cd e0
    # fg hi j0
    # kl mn o0
    # 00 00 00
    # 00 00 00
    # 00 00 00
    shape = full_shape([frames + segments, segments, frame_step])
    signal = paddle.reshape(signal, shape)

    # Transpose dimensions so that signal.shape = (3, 6, 2)
    # ab fg kl 00 00 00
    # cd hi mn 00 00 00
    # e0 j0 o0 00 00 00
    perm = list(range(outer_rank)) + [1 + outer_rank, outer_rank, 2 + outer_rank]
    signal = paddle.transpose(signal, perm=perm, name=None)

    # Reshape so that signal.shape = (18, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0 00 00 00
    shape = full_shape([(frames + segments) * segments, frame_step])
    signal = paddle.reshape(signal, shape)

    # Truncate so that signal.shape = (15, 2)
    # ab fg kl 00 00 00 cd hi mn 00 00 00 e0 j0 o0
    signal = signal[..., : (frames + segments - 1) * segments, :]

    # Reshape so that signal.shape = (3, 5, 2)
    # ab fg kl 00 00
    # 00 cd hi mn 00
    # 00 00 e0 j0 o0
    shape = full_shape([segments, frames + segments - 1, frame_step])
    signal = paddle.reshape(signal, shape)

    # Now, reduce over the columns, to achieve the desired sum.
    signal = paddle.sum(signal, axis=-3)

    # Flatten the array.
    shape = full_shape([(frames + segments - 1) * frame_step])
    signal = paddle.reshape(signal, shape)

    # Truncate to final length.
    signal = signal[..., :output_size]

    # Paddle Error
    # RuntimeError: (NotFound) The kernel with key (CPU, NCHW, float16) of kernel
    # `pad` is not registered.
    if signal_dtype is not None:
        signal = signal.astype(signal_dtype)

    return signal
