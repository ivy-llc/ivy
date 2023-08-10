# global

import math
import numpy as np
from typing import Optional, Union, Tuple, List, Literal, Sequence

# local
import ivy
from ivy.functional.ivy.layers import (
    _handle_padding,
    _get_num_padded_values,
    _validate_max_pool_params,
    _depth_max_pooling_helper,
)
from ivy.functional.backends.numpy.layers import _add_dilations
from ivy.functional.ivy.experimental.layers import _padding_ceil_mode
from ivy.func_wrapper import with_supported_dtypes
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _determine_depth_max_pooling(x, kernel, strides, dims, data_format="channel_last"):
    kernel, strides, depth_pooling = _depth_max_pooling_helper(
        x.shape, kernel, strides, dims=dims, data_format=data_format
    )
    if depth_pooling:
        x = np.transpose(x, (0, dims + 1, *range(1, dims + 1)))
    return x, kernel, strides, depth_pooling


def max_pool1d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dims = 1
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims=dims
    )

    if data_format == "NCW":
        x = np.swapaxes(x, 1, 2)
        kernel = [kernel[i] for i in [0, 2, 1]] if len(kernel) == (dims + 2) else kernel
        strides = (
            [strides[i] for i in [0, 2, 1]] if len(strides) == (dims + 2) else strides
        )
        padding = (
            [padding[i] for i in [0, 2, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )
        padding = (
            [padding[i] for i in [0, 2, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_last"
    )

    x_shape = x.shape[1:2]
    filters = np.ones((list(kernel)), dtype=x.dtype)
    if not depth_pooling:
        if dilation[0] > 1:
            filters = _add_dilations(filters, dilation[0], axis=0, values=0)
        kernel = list(filters.shape)
        pad_list = padding
        if isinstance(padding, str):
            pad_w = _handle_padding(x_shape[0], strides[0], kernel[0], padding)
            pad_list = [
                (pad_w // 2, pad_w - pad_w // 2),
            ]
        if ceil_mode:
            pad_list[0] = _padding_ceil_mode(
                x_shape[0], kernel[0], pad_list[0], strides[0]
            )

        x = np.pad(
            x,
            [
                (0, 0),
                *pad_list,
                (0, 0),
            ],
            "constant",
            constant_values=-math.inf,
        )
    else:
        if isinstance(padding, list) and any(
            [item != 0 for sublist in padding for item in sublist]
        ):
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )

    x_shape = x.shape
    new_w = (x_shape[1] - kernel[0]) // strides[0] + 1
    new_shape = [x_shape[0], new_w] + list(kernel) + [x_shape[-1]]
    new_strides = (
        x.strides[0],
        x.strides[1] * strides[0],
        x.strides[1],
        x.strides[2],
    )

    # B x OW x KW x I
    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    # B x OW x KW x I
    sub_matrices = np.where(
        filters.reshape([1] * 2 + list(kernel) + [1]), sub_matrices, -math.inf
    )

    res = sub_matrices.max(axis=(2))

    if depth_pooling:
        res = np.swapaxes(res, 1, 2)
    if data_format == "NCW":
        res = np.swapaxes(res, 1, 2)
    return res


def max_pool2d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dims = 2
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims=dims
    )

    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        kernel = (
            [kernel[i] for i in [0, 2, 3, 1]] if len(kernel) == (dims + 2) else kernel
        )
        strides = (
            [strides[i] for i in [0, 2, 3, 1]]
            if len(strides) == (dims + 2)
            else strides
        )
        padding = (
            [padding[i] for i in [0, 2, 3, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_last"
    )

    x_shape = list(x.shape[1:3])
    filters = np.ones((list(kernel)), dtype=x.dtype)
    if not depth_pooling:
        for j in range(dims):
            if dilation[j] > 1:
                filters = _add_dilations(filters, dilation[j], axis=j, values=0)
        kernel = list(filters.shape)
        pad_list = padding
        if isinstance(padding, str):
            pad_h = _handle_padding(x_shape[0], strides[0], kernel[0], padding)
            pad_w = _handle_padding(x_shape[1], strides[1], kernel[1], padding)
            pad_list = [
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ]
        pad_list = list(pad_list)
        if ceil_mode:
            for i in range(dims):
                pad_list[i] = _padding_ceil_mode(
                    x_shape[i], kernel[i], pad_list[i], strides[i]
                )

        x = np.pad(
            x,
            [
                (0, 0),
                *pad_list,
                (0, 0),
            ],
            "constant",
            constant_values=-math.inf,
        )
    else:
        if isinstance(padding, list) and any(
            [item != 0 for sublist in padding for item in sublist]
        ):
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
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

    # B x OH x OW x KH x KW x I
    sub_matrices = np.where(
        filters.reshape([1] * 3 + list(kernel) + [1]), sub_matrices, -math.inf
    )

    # B x OH x OW x O
    res = sub_matrices.max(axis=(3, 4))

    if depth_pooling:
        res = np.transpose(res, (0, 2, 3, 1))
    if data_format == "NCHW":
        return np.transpose(res, (0, 3, 1, 2))
    return res


def max_pool3d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    dims = 3
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims=dims
    )

    if data_format == "NCDHW":
        x = np.transpose(x, (0, 2, 3, 4, 1))
        kernel = (
            [kernel[i] for i in [0, 2, 3, 4, 1]]
            if len(kernel) == (dims + 2)
            else kernel
        )
        strides = (
            [strides[i] for i in [0, 2, 3, 4, 1]]
            if len(strides) == (dims + 2)
            else strides
        )
        padding = (
            [padding[i] for i in [0, 2, 3, 4, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_last"
    )

    x_shape = x.shape[1:4]
    filters = np.ones((list(kernel)), dtype=x.dtype)
    if not depth_pooling:
        for j in range(dims):
            if dilation[j] > 1:
                filters = _add_dilations(filters, dilation[j], axis=j, values=0)
        kernel = list(filters.shape)
        pad_list = padding
        if isinstance(padding, str):
            pad_d = _handle_padding(x_shape[0], strides[0], kernel[0], padding)
            pad_h = _handle_padding(x_shape[1], strides[1], kernel[1], padding)
            pad_w = _handle_padding(x_shape[2], strides[2], kernel[2], padding)
            pad_list = [
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ]
        pad_list = list(pad_list)
        if ceil_mode:
            for i in range(dims):
                pad_list[i] = _padding_ceil_mode(
                    x_shape[i], kernel[i], pad_list[i], strides[i]
                )

        x = np.pad(
            x,
            [
                (0, 0),
                *pad_list,
                (0, 0),
            ],
            "constant",
            constant_values=-math.inf,
        )
    else:
        if isinstance(padding, list) and any(
            [item != 0 for sublist in padding for item in sublist]
        ):
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
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
    # B x OD x OH x OW x KD x KH x KW x I
    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )

    # B x OD x OH x OW x KD x KH x KW x I
    sub_matrices = np.where(
        filters.reshape([1] * 4 + list(kernel) + [1]), sub_matrices, -math.inf
    )

    # B x OD x OH x OW x O
    res = sub_matrices.max(axis=(4, 5, 6))

    if depth_pooling:
        res = np.transpose(res, (0, 2, 3, 4, 1))
    if data_format == "NCDHW":
        return np.transpose(res, (0, 4, 1, 2, 3))
    return res


def _get_padded_values(x_shape, kernel, strides, padding, ceil_mode, dim):
    if isinstance(padding, str):
        pad_specific = [
            _handle_padding(x_shape[i], strides[i], kernel[i], padding)
            for i in range(dim)
        ]
        padding = [
            (pad_specific[i] // 2, pad_specific[i] - pad_specific[i] // 2)
            for i in range(dim)
        ]
    else:
        pad_specific = [sum(padding[i]) for i in range(dim)]

    c = []
    if ceil_mode:
        for i in range(dim):
            padding[i], c_i = _padding_ceil_mode(
                x_shape[i], kernel[i], padding[i], strides[i], True
            )
            c.append(c_i)
            pad_specific[i] = sum(padding[i])
    return padding, pad_specific, c


def avg_pool1d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
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

    if data_format in ("NCW", "NCL"):
        x = np.swapaxes(x, 1, 2)

    x_shape = x.shape[1:-1]
    padding, pad_specific, c = _get_padded_values(
        x_shape, kernel, strides, padding, ceil_mode, 1
    )

    x = np.pad(
        x,
        [
            (0, 0),
            *padding,
            (0, 0),
        ],
        constant_values=0.0,
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

    if (not count_include_pad or ceil_mode) and any(pad_specific):
        if not count_include_pad:
            num_padded_values = np.array(
                ivy.map(
                    _get_num_padded_values,
                    constant={
                        "p": pad_specific[0],
                        "n": x.shape[1] - pad_specific[0],
                        "k": kernel[0],
                        "s": strides[0],
                    },
                    unique={
                        "i": np.arange(res.shape[1]),
                    },
                ),
                dtype=res.dtype,
            )
        else:
            num_padded_values = np.zeros(res.shape[1], dtype=res.dtype)
            num_padded_values[-1] = c[0]
        res = (kernel[0] * res) / (kernel[0] - num_padded_values[:, None])

    if data_format in ("NCW", "NCL"):
        return res.swapaxes(1, 2)

    return res


def avg_pool2d(
    x: np.ndarray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
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
    padding, pad_specific, c = _get_padded_values(
        x_shape, kernel, strides, padding, ceil_mode, 2
    )
    x = np.pad(
        x,
        [
            (0, 0),
            *padding,
            (0, 0),
        ],
        constant_values=0.0,
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
    if divisor_override is not None:
        res = np.sum(sub_matrices, axis=(3, 4)) / divisor_override
    else:
        res = np.mean(sub_matrices, axis=(3, 4))
    if (
        (not count_include_pad or ceil_mode)
        and any(pad_specific)
        and not divisor_override
    ):
        if not count_include_pad:
            num_padded_values = [
                np.array(
                    ivy.map(
                        _get_num_padded_values,
                        constant={
                            "p": pad_specific[i],
                            "n": x.shape[i + 1] - pad_specific[i],
                            "k": kernel[i],
                            "s": strides[i],
                        },
                        unique={
                            "i": np.arange(res.shape[i + 1]),
                        },
                    ),
                    dtype=res.dtype,
                )
                for i in range(2)
            ]
        else:
            num_padded_values = []
            for i in range(2):
                num_pad = np.zeros(res.shape[i + 1], dtype=res.dtype)
                num_pad[-1] = c[i]
                num_padded_values.append(num_pad)
        num_padded_values1 = num_padded_values[0][:, None]
        num_padded_values2 = num_padded_values[1][None, :]
        num_padded_values = (
            num_padded_values1 * kernel[1]
            + num_padded_values2 * kernel[0]
            - num_padded_values1 * num_padded_values2
        )
        kernel_mul = np.prod(kernel)
        res = (kernel_mul * res) / (kernel_mul - np.expand_dims(num_padded_values, -1))

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
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
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
    padding, pad_specific, c = _get_padded_values(
        x_shape, kernel, strides, padding, ceil_mode, 3
    )

    x = np.pad(
        x,
        [
            (0, 0),
            *padding,
            (0, 0),
        ],
        constant_values=0.0,
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
    if divisor_override is not None:
        res = np.sum(sub_matrices, axis=(4, 5, 6)) / divisor_override
    else:
        res = np.mean(sub_matrices, axis=(4, 5, 6))

    if (
        (not count_include_pad or ceil_mode)
        and any(pad_specific)
        and not divisor_override
    ):
        if not count_include_pad:
            num_padded_values = [
                np.array(
                    ivy.map(
                        _get_num_padded_values,
                        constant={
                            "p": pad_specific[i],
                            "n": x.shape[i + 1] - pad_specific[i],
                            "k": kernel[i],
                            "s": strides[i],
                        },
                        unique={
                            "i": np.arange(res.shape[i + 1]),
                        },
                    ),
                    dtype=res.dtype,
                )
                for i in range(3)
            ]
        else:
            num_padded_values = []
            for i in range(3):
                num_pad = np.zeros(res.shape[i + 1], dtype=res.dtype)
                num_pad[-1] = c[i]
                num_padded_values.append(num_pad)
        num_padded_values1 = num_padded_values[0].reshape((-1, 1, 1))
        num_padded_values2 = num_padded_values[1].reshape((1, -1, 1))
        num_padded_values3 = num_padded_values[2].reshape((1, 1, -1))
        num_padded_values = (
            num_padded_values1 * kernel[1] * kernel[2]
            + num_padded_values2 * kernel[0] * kernel[2]
            + num_padded_values3 * kernel[0] * kernel[1]
            + num_padded_values1 * num_padded_values2 * num_padded_values3
            - num_padded_values1 * num_padded_values2 * kernel[2]
            - num_padded_values1 * num_padded_values3 * kernel[1]
            - num_padded_values2 * num_padded_values3 * kernel[0]
        )
        kernel_mul = np.prod(kernel)
        res = (kernel_mul * res) / (kernel_mul - np.expand_dims(num_padded_values, -1))
    if data_format == "NCDHW":
        return np.transpose(res, (0, 4, 1, 2, 3))
    return res


def fft(
    x: np.ndarray,
    dim: int,
    /,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not isinstance(dim, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(dim)}"
        )
    if n is None:
        n = x.shape[dim]
    if n < -len(x.shape):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n <= 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {n}, expecting more than 1"
        )
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    if x.dtype in [np.uint64, np.int64, np.float64, np.complex128]:
        out_dtype = np.complex128
    else:
        out_dtype = np.complex64
    return np.fft.fft(x, n, dim, norm).astype(out_dtype)


@with_supported_dtypes({"1.25.2 and below": ("float32", "float64")}, backend_version)
def dct(
    x: np.ndarray,
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
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


def idct(
    x: np.ndarray,
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return dct(x, type=inverse_type, n=n, axis=axis, norm=norm, out=out)


def dropout1d(
    x: np.ndarray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if training:
        x_shape = x.shape
        is_batched = len(x_shape) == 3
        if data_format == "NCW":
            perm = (0, 2, 1) if is_batched else (1, 0)
            x = np.transpose(x, perm)
            x_shape = x.shape
        mask = np.random.binomial(1, 1 - prob, x_shape)
        res = np.where(mask, x / (1 - prob), 0)
        if data_format == "NCW":
            res = np.transpose(res, perm)
    else:
        res = x
    return res


def dropout2d(
    x: np.ndarray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if training:
        x_shape = x.shape
        is_batched = len(x_shape) == 4
        if data_format == "NCHW":
            perm = (0, 2, 3, 1) if is_batched else (1, 2, 0)
            x = np.transpose(x, perm)
            x_shape = x.shape
        mask = np.random.binomial(1, 1 - prob, x_shape)
        res = np.where(mask, x / (1 - prob), 0)
        if data_format == "NCHW":
            perm = (0, 3, 1, 2) if is_batched else (2, 0, 1)
            res = np.transpose(res, perm)
    else:
        res = x
    return res


def dropout3d(
    x: np.ndarray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if training:
        x_shape = x.shape
        is_batched = len(x_shape) == 5
        if data_format == "NCDHW":
            perm = (0, 2, 3, 4, 1) if is_batched else (1, 2, 3, 0)
            x = np.transpose(x, perm)
            x_shape = x.shape
        mask = np.random.binomial(1, 1 - prob, x_shape)
        res = np.where(mask, x / (1 - prob), 0)
        if data_format == "NCDHW":
            perm = (0, 4, 1, 2, 3) if is_batched else (3, 0, 1, 2)
            res = np.transpose(res, perm)
    else:
        res = x
    return res


def ifft(
    x: np.ndarray,
    dim: int,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not isinstance(dim, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(dim)}"
        )
    if n is None:
        n = x.shape[dim]
    if n < -len(x.shape):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n <= 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {n}, expecting more than 1"
        )
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return np.asarray(np.fft.ifft(x, n, dim, norm), dtype=x.dtype)


def stft(
    signal: Union[np.ndarray, int, Tuple[int]],
    n_fft: Union[int, Tuple[int]],
    frame_step: int,
    /,
    *,
    axis: Optional[int] = None,
    onesided:Optional[bool] = True,
    fs: Optional[float] = 1.0,
    window: Optional[Union[np.ndarray, list, str, Tuple[int]]] = None,
    win_length: Optional[int] = None,
    nperseg: Optional[int] = 256,
    noverlap: Optional[int] = None,
    center: Optional[bool] = True,
    pad_mode: Optional[str] = "reflect",
    normalized: Optional[bool] = False,
    detrend: Optional[Union[str, callable, bool]] = False,
    return_complex: Optional[bool] = True,
    boundary: Optional[str] = 'zeros',
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if window is None:
        window = np.hanning(n_fft)
    else:
        window = np.asarray(window)

    if center:
        pad = (n_fft - frame_step) // 2
        signal = np.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(pad, pad)], mode=pad_mode)

    num_frames = (signal.shape[-1] - n_fft) // frame_step + 1
    stft_result = np.empty(signal.shape[:-1] + (num_frames, n_fft // 2 + 1), dtype=np.complex128)

    for i in range(num_frames):
        start = i * frame_step
        end = start + n_fft
        frame = signal[..., start:end]

        if win_length is not None:
            frame = frame * window[:len(frame)]
        else:
            frame = frame * window

        stft_frame = np.fft.fft(frame, n=n_fft, axis=-1)
        if onesided:
            stft_frame = stft_frame[..., :n_fft // 2 + 1]

        if detrend:
            detrend_func = np.poly1d if isinstance(detrend, bool) else detrend
            detrended = detrend_func(np.arange(len(frame)))(frame)
            stft_frame = np.fft.fft(detrended, n=n_fft, axis=-1)
            if onesided:
                stft_frame = stft_frame[..., :n_fft // 2 + 1]

        stft_result[..., i, :n_fft // 2 + 1] = stft_frame

    return stft_result
        
    
def fft2(
    x: np.ndarray,
    *,
    s: Sequence[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ivy.utils.assertions.check_elem_in_list(
        norm,
        ["backward", "ortho", "forward"],
        message=f"Unrecognized normalization mode {norm}",
    )
    if not all(isinstance(j, int) for j in dim):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {dim} to be a sequence of integers <class integer>"
        )
    if s is None:
        s = (x.shape[dim[0]], x.shape[dim[1]])
    if all(j < -len(x.shape) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not all(isinstance(j, int) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {s} to be a sequence of integers <class integer>"
        )
    if all(j <= 1 for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {s}, expecting s points larger than 1"
        )
    return np.fft.fft2(x, s, dim, norm).astype(np.complex128)


def ifftn(
    x: np.ndarray,
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: str = "backward",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fft.ifftn(x, s, axes, norm).astype(x.dtype)


@with_unsupported_dtypes({"1.25.2 and below": ("complex",)}, backend_version)
def embedding(
    weights: np.ndarray,
    indices: np.ndarray,
    /,
    *,
    max_norm: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ivy.utils.assertions.check_equal(
        len(weights.shape), 2, message="weights must be 2-d", as_array=False
    )

    embeddings = np.take(weights, indices, axis=0)
    if max_norm is not None:
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = np.where(
            norms > max_norm, embeddings * max_norm / norms, embeddings
        )
        embeddings = np.where(
            norms < -max_norm, embeddings * -max_norm / norms, embeddings
        )
    return embeddings


def rfftn(
    x: np.ndarray,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    *,
    norm: str = "backward",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not all(isinstance(j, int) for j in axes):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {axes} to be a sequence of integers <class integer>"
        )
    if s is None:
        s = (x.shape[axes[0]], x.shape[axes[1]])
    if all(j < -len(x.shape) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {axes}, expecting ranging"
            f" from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not all(isinstance(j, int) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {s} to be a sequence of integers <class integer>"
        )
    if all(j <= 1 for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {s}, expecting s points larger than 1"
        )
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return np.fft.rfftn(x, s, axes, norm).astype(np.complex128)
