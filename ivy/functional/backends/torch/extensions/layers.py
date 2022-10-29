# global
from typing import Optional, Union, Tuple, Literal
import torch
import math

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def flatten(
    x: torch.Tensor,
    /,
    *,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)


def vorbis_window(
    window_length: torch.tensor,
    *,
    dtype: Optional[torch.dtype] = torch.float32,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.tensor(
        [
            round(
                math.sin(
                    (ivy.pi / 2) * (math.sin(ivy.pi * (i) / (window_length * 2)) ** 2)
                ),
                8,
            )
            for i in range(1, window_length * 2)[0::2]
        ],
        dtype=dtype,
    )


vorbis_window.support_native_out = False


def hann_window(
    window_length: int,
    periodic: Optional[bool] = True,
    dtype: Optional[torch.dtype] = None,
    *,
    out: Optional[torch.tensor] = None,
) -> torch.tensor:
    return torch.hann_window(
        window_length,
        periodic=periodic,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=None,
    )


hann_window.support_native_out = False


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def max_pool2d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0])

    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x_shape = list(x.shape[2:])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    x = torch.nn.functional.pad(
        x,
        [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
        value=float("-inf"),
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.max_pool2d(x, kernel, strides, 0)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def max_pool1d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides,)
    elif len(strides) == 1:
        strides = (strides[0],)

    if isinstance(kernel, int):
        kernel = (kernel,)
    elif len(kernel) == 1:
        kernel = (kernel[0],)

    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    x_shape = x.shape[2]
    pad_w = ivy.handle_padding(x_shape, strides[0], kernel[0], padding)
    x = torch.nn.functional.pad(
        x, [pad_w // 2, pad_w - pad_w // 2], value=float("-inf")
    )

    res = torch.nn.functional.max_pool1d(x, kernel, strides, 0)

    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.kaiser_window(
        window_length,
        periodic,
        beta,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    )


def hamming_window(
    window_length: int,
    /,
    *,
    periodic: Optional[bool] = True,
    alpha: Optional[float] = 0.54,
    beta: Optional[float] = 0.46,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.hamming_window(
        window_length,
        periodic=periodic,
        alpha=alpha,
        beta=beta,
        dtype=dtype,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    )


def dct(
    x: torch.Tensor,
    /,
    *,
    type: Optional[Literal[1, 2, 3, 4]] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.tensor:
    if norm not in (None, "ortho"):
        raise ValueError("Norm must be either None or 'ortho'")
    if x.dtype not in [torch.float32, torch.float64]:
        x = x.type(torch.float32)
    if axis != -1:
        x = torch.transpose(x, axis, -1)
    if n is not None:
        signal_len = x.shape[-1]
        if n <= signal_len:
            x = x[..., :n]
        else:
            x = torch.nn.functional.pad(x, (0, n - signal_len))
    real_zero = torch.tensor(0.0, dtype=x.dtype)
    axis_dim = x.shape[-1]
    axis_dim_float = torch.tensor(axis_dim, dtype=x.dtype)

    if type == 1:
        if norm:
            raise ValueError("Normalization not supported for type-I DCT")
        x = torch.concat([x, x.flip(-1)[..., 1:-1]], dim=-1)
        dct_out = torch.real(torch.fft.rfft(x, dim=-1))

    elif type == 2:
        scale = 2.0 * torch.exp(
            torch.complex(
                real_zero,
                -torch.arange(axis_dim_float) * math.pi * 0.5 / axis_dim_float,
            )
        )
        dct_out = torch.real(torch.fft.rfft(x, n=2 * axis_dim)[..., :axis_dim] * scale)
        if norm == "ortho":
            n1 = 0.5 * torch.rsqrt(axis_dim_float)
            n2 = n1 * math.sqrt(2.0)
            # vectorising scaling factors
            sf = torch.nn.functional.pad(n1.unsqueeze(0), (0, axis_dim - 1), value=n2)
            dct_out = sf * dct_out

    elif type == 3:
        if norm == "ortho":
            n1 = torch.sqrt(axis_dim_float)
            n2 = n1 * math.sqrt(0.5)
            sf = torch.nn.functional.pad(n1.unsqueeze(0), (0, axis_dim - 1), value=n2)
            x = x * sf
        else:
            x = x * axis_dim_float
        scale = 2.0 * torch.exp(
            torch.complex(
                real_zero, torch.arange(axis_dim_float) * math.pi * 0.5 / axis_dim_float
            )
        )
        dct_out = torch.real(
            torch.fft.irfft(scale * torch.complex(x, real_zero), n=2 * axis_dim)
        )[..., :axis_dim]

    elif type == 4:
        dct_2 = dct(x, type=2, n=2 * axis_dim, norm=None)
        dct_out = dct_2[..., 1::2]
        if norm == "ortho":
            dct_out *= math.sqrt(0.5) * torch.rsqrt(axis_dim_float)

    if axis != -1:
        dct_out = torch.transpose(dct_out, axis, -1)
    return dct_out
