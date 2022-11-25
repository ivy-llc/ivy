# global

import math
import numpy as np
from typing import Optional, Union, Tuple, Literal

# local
import ivy


def max_pool1d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    if isinstance(kernel, int):
        kernel = [kernel]
    elif len(kernel) == 1:
        kernel = [kernel[0]]

    if isinstance(strides, int):
        strides = [strides]
    elif len(strides) == 1:
        strides = [strides[0]]

    if data_format == "NCW":
        x = x.permute(0, 2, 1)

    pad_w = ivy.handle_padding(x.shape[1], strides[0], kernel[0], padding)
    x = np.pad(
        x,
        [
            (0, 0),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ],
        "edge",
    )

    x_shape = x.shape
    new_w = (x_shape[1] - kernel[0]) // strides[0] + 1
    new_shape = [x_shape[0], new_w, kernel[0]] + [x_shape[-1]]
    new_strides = (
        x.strides[0],
        x.strides[1] * strides[0],
        x.strides[1],
        x.strides[2],
    )

    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    res = sub_matrices.max(axis=(2))

    if data_format == "NCW":
        return res.permute(0, 2, 1)
    return res


def max_pool2d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    if isinstance(kernel, int):
        kernel = [kernel] * 2
    elif len(kernel) == 1:
        kernel = [kernel[0]] * 2

    if isinstance(strides, int):
        strides = [strides] * 2
    elif len(strides) == 1:
        strides = [strides[0]] * 2

    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))

    x_shape = list(x.shape[1:3])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    x = np.pad(
        x,
        [
            (0, 0),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ],
        "edge",
    )

    x_shape = x.shape
    new_h = (x_shape[1] - kernel[0]) // strides[0] + 1
    new_w = (x_shape[2] - kernel[1]) // strides[1] + 1
    new_shape = [x_shape[0], new_h, new_w] + list(kernel) + [x_shape[-1]]
    new_strides = (
        x.strides[0],
        x.strides[1] * strides[1],
        x.strides[2] * strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[3],
    )
    # B x OH x OW x KH x KW x I
    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    # B x OH x OW x O
    res = sub_matrices.max(axis=(3, 4))
    if data_format == "NCHW":
        return np.transpose(res, (0, 3, 1, 2))
    return res


def max_pool3d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    if isinstance(kernel, int):
        kernel = [kernel] * 3
    elif len(kernel) == 1:
        kernel = [kernel[0]] * 3

    if isinstance(strides, int):
        strides = [strides] * 3
    elif len(strides) == 1:
        strides = [strides[0]] * 3

    if data_format == "NCDHW":
        x = np.transpose(x, (0, 2, 3, 4, 1))

    x_shape = list(x.shape[1:4])
    pad_d = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_h = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    pad_w = ivy.handle_padding(x_shape[2], strides[2], kernel[2], padding)

    x = np.pad(
        x,
        [
            (0, 0),
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ],
        "edge",
    )

    x_shape = x.shape
    new_d = (x_shape[1] - kernel[0]) // strides[0] + 1
    new_h = (x_shape[2] - kernel[1]) // strides[1] + 1
    new_w = (x_shape[3] - kernel[2]) // strides[2] + 1
    new_shape = [x_shape[0], new_d, new_h, new_w] + list(kernel) + [x_shape[-1]]
    new_strides = (
        x.strides[0],
        x.strides[1] * strides[0],
        x.strides[2] * strides[1],
        x.strides[3] * strides[2],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        x.strides[4],
    )
    # B x OH x OW x KH x KW x I
    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    # B x OH x OW x O
    res = sub_matrices.max(axis=(4, 5, 6))
    if data_format == "NCDHW":
        return np.transpose(res, (0, 4, 1, 2, 3))
    return res


def avg_pool1d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    if isinstance(kernel, int):
        kernel = [kernel]
    elif len(kernel) == 1:
        kernel = [kernel[0]]

    if isinstance(strides, int):
        strides = [strides]
    elif len(strides) == 1:
        strides = [strides[0]]

    if data_format == "NCW":
        x = x.permute(0, 2, 1)

    pad_w = ivy.handle_padding(x.shape[1], strides[0], kernel[0], padding)
    x = np.pad(
        x,
        [
            (0, 0),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ],
        "edge",
    )

    x_shape = x.shape
    new_w = (x_shape[1] - kernel[0]) // strides[0] + 1
    new_shape = [x_shape[0], new_w, kernel[0]] + [x_shape[-1]]
    new_strides = (
        x.strides[0],
        x.strides[1] * strides[0],
        x.strides[1],
        x.strides[2],
    )

    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    res = np.mean(sub_matrices, axis=2)

    if data_format == "NCW":
        return res.permute(0, 2, 1)
    return res


def avg_pool2d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(strides, int):
        strides = [strides] * 2
    elif len(strides) == 1:
        strides = [strides[0]] * 2

    if isinstance(strides, int):
        strides = [strides] * 2
    elif len(strides) == 1:
        strides = [strides[0]] * 2

    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))

    x_shape = list(x.shape[1:3])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    x = np.pad(
        x,
        [
            (0, 0),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ],
        "edge",
    )

    x_shape = x.shape
    new_h = (x_shape[1] - kernel[0]) // strides[0] + 1
    new_w = (x_shape[2] - kernel[1]) // strides[1] + 1
    new_shape = [x_shape[0], new_h, new_w] + list(kernel) + [x_shape[-1]]
    new_strides = (
        x.strides[0],
        x.strides[1] * strides[0],
        x.strides[2] * strides[1],
        x.strides[1],
        x.strides[2],
        x.strides[3],
    )
    # B x OH x OW x KH x KW x I
    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    # B x OH x OW x O
    res = np.mean(sub_matrices, axis=(3, 4))
    if data_format == "NCHW":
        return np.transpose(res, (0, 3, 1, 2))
    return res


def avg_pool3d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    if isinstance(kernel, int):
        kernel = [kernel] * 3
    elif len(kernel) == 1:
        kernel = [kernel[0]] * 3

    if isinstance(strides, int):
        strides = [strides] * 3
    elif len(strides) == 1:
        strides = [strides[0]] * 3

    if data_format == "NCDHW":
        x = np.transpose(x, (0, 2, 3, 4, 1))

    x_shape = list(x.shape[1:4])
    pad_d = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_h = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    pad_w = ivy.handle_padding(x_shape[2], strides[2], kernel[2], padding)

    x = np.pad(
        x,
        [
            (0, 0),
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ],
        "edge",
    )

    x_shape = x.shape
    new_d = (x_shape[1] - kernel[0]) // strides[0] + 1
    new_h = (x_shape[2] - kernel[1]) // strides[1] + 1
    new_w = (x_shape[3] - kernel[2]) // strides[2] + 1
    new_shape = [x_shape[0], new_d, new_h, new_w] + list(kernel) + [x_shape[-1]]
    new_strides = (
        x.strides[0],
        x.strides[1] * strides[0],
        x.strides[2] * strides[1],
        x.strides[3] * strides[2],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        x.strides[4],
    )
    # B x OH x OW x KH x KW x I
    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    # B x OH x OW x O
    res = np.mean(sub_matrices, axis=(4, 5, 6))
    if data_format == "NCDHW":
        return np.transpose(res, (0, 4, 1, 2, 3))
    return res


def fft(
    x: np.ndarray,
    dim: int,
    /,
    *,
    norm: Optional[str] = "backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not isinstance(dim, int):
        raise ivy.exceptions.IvyError(f"Expecting <class 'int'> instead of {type(dim)}")
    if n is None:
        n = x.shape[dim]
    if n < -len(x.shape):
        raise ivy.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not isinstance(n, int):
        raise ivy.exceptions.IvyError(f"Expecting <class 'int'> instead of {type(n)}")
    if n <= 1:
        raise ivy.exceptions.IvyError(f"Invalid data points {n}, expecting more than 1")
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return np.fft.fft(x, n, dim, norm)


def dct(
    x: np.ndarray,
    /,
    *,
    type: Optional[Literal[1, 2, 3, 4]] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if norm not in (None, "ortho"):
        raise ValueError("Norm must be either None or 'ortho'")
    if axis < 0:
        axis = axis + len(x.shape)
    if n is not None:
        signal_len = x.shape[axis]
        if n <= signal_len:
            local_idx = [slice(None)] * len(x.shape)
            local_idx[axis] = slice(None, n)
            x = x[tuple(local_idx)]
        else:
            pad_idx = [[0, 0] for _ in range(len(x.shape))]
            pad_idx[axis][1] = n - signal_len
            x = np.pad(x, pad_idx)
    real_zero = np.array(0.0, dtype=x.dtype)
    axis_dim = x.shape[axis]
    axis_dim_float = np.array(axis_dim, dtype=x.dtype)
    cast_final = True if x.dtype != np.float64 else False

    if type == 1:
        if norm:
            raise ValueError("Normalization not supported for type-I DCT")
        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(-2, 0, -1)
        x = np.concatenate([x, x[tuple(axis_idx)]], axis=axis)
        dct_out = np.real(np.fft.rfft(x, axis=axis))

    elif type == 2:
        cmplx = np.empty(axis_dim, dtype=np.complex64)
        cmplx.real = real_zero
        cmplx.imag = -np.arange(axis_dim_float) * math.pi * 0.5 / axis_dim_float

        scale_dims = [1] * len(x.shape)
        scale_dims[axis] = axis_dim
        scale = 2.0 * np.exp(cmplx).reshape(scale_dims)

        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(None, axis_dim)
        dct_out = np.real(
            np.fft.rfft(x, n=2 * axis_dim, axis=axis)[tuple(axis_idx)] * scale
        )

        if norm == "ortho":
            n1 = 0.5 * np.reciprocal(np.sqrt(axis_dim_float))
            n2 = n1 * math.sqrt(2.0)
            sf = np.pad(np.expand_dims(n1, 0), (0, axis_dim - 1), constant_values=n2)
            dct_out = sf.reshape(scale_dims) * dct_out

    elif type == 3:
        cmplx = np.empty(axis_dim, dtype=np.complex64)
        cmplx.real = real_zero
        cmplx.imag = np.arange(axis_dim_float) * math.pi * 0.5 / axis_dim_float

        scale_dims = [1] * len(x.shape)
        scale_dims[axis] = axis_dim
        scale = 2.0 * np.exp(cmplx).reshape(scale_dims)

        if norm == "ortho":
            n1 = np.sqrt(axis_dim_float)
            n2 = n1 * np.sqrt(0.5)
            sf = np.pad(np.expand_dims(n1, 0), (0, axis_dim - 1), constant_values=n2)
            x = x * sf.reshape(scale_dims)
        else:
            x = x * axis_dim_float

        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(None, axis_dim)

        x = x.astype(np.complex64)
        x.imag = real_zero
        dct_out = np.real(np.fft.irfft(scale * x, n=2 * axis_dim, axis=axis))[
            tuple(axis_idx)
        ]

    elif type == 4:
        dct_2 = dct(x, type=2, n=2 * axis_dim, axis=axis, norm=None)
        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(1, None, 2)
        dct_out = dct_2[tuple(axis_idx)]
        if norm == "ortho":
            dct_out *= math.sqrt(0.5) * np.reciprocal(np.sqrt(axis_dim_float))

    return dct_out.astype(np.float32) if cast_final else dct_out
