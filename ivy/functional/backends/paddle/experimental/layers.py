# global
from typing import Optional, Union, Tuple, Literal, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

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
    raise IvyNotImplementedException()


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


def stft(
        x: paddle.Tensor,
        input: paddle.Tensor,
        signal: Union[paddle.Tensor, paddle.Variable],
        frame_step: Union[int, Tuple[int]],
        n_fft: Union[int, Tuple[int]],
        /,
        *,
        axis: int = 1,
        onesided:Optional[bool] = False,
        fs: Optional[float] = 1.0,
        hop_length: Optional[Union[int, Tuple[int]]] = None,
        win_length: Optional[Union[int, Tuple[int]]] = None,
        dft_length: Optional[Union[int, Tuple[int]]] = None,
        window: Optional[Union[paddle.Tensor, str, int, Tuple[int]]] = None,
        frame_length: Optional[Union[int, Tuple[int]]] = None,
        nperseg: Optional[int] = 256,
        noverlap: Optional[int] = None,
        center: Optional[bool] = True,
        pad_mode: Optional[str] = "reflect",
        normalized: Optional[bool] = False,
        nfft: Optional[int] = None,
        detrend: Optional[str] = None,
        return_onesided: Optional[bool] = True,
        return_complex: Optional[bool] = True,
        boundary: Optional[str] = None,
        name: Optional[str] = None,
        padded: Optional[bool] = True,
        out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.signal.stft(
        x,
        input,
        signal,
        axis,
        frame_step,
        n_fft,
        onesided,
        fs,
        hop_length,
        win_length,
        dft_length,
        window,
        frame_length,
        nperseg,
        noverlap,
        center,
        pad_mode,
        normalized,
        nfft,
        detrend,
        return_onesided,
        return_complex,
        boundary,
        padded,
        name,
        norm,
        out,
    )
    