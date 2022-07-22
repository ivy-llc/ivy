"""Collection of Numpy network layers, wrapped to fit Ivy syntax and signature."""

# global
import numpy as np
from typing import Union, Tuple, Optional


def conv1d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: int,
    padding: str,
    data_format: str = "NWC",
    dilations: int = 1,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    if isinstance(strides, tuple):
        strides = strides[0]
    if isinstance(dilations, tuple):
        dilations = dilations[0]
    if data_format == "NCW":
        x = np.transpose(x, (0, 2, 1))
    x_shape = (1,) + x.shape
    filter_shape = (1,) + filters.shape
    x_strides = (x.strides[0],) + x.strides
    filter_strides = (filters.strides[0],) + filters.strides
    x = np.lib.stride_tricks.as_strided(x, shape=x_shape, strides=x_strides)
    filters = np.lib.stride_tricks.as_strided(
        filters, shape=filter_shape, strides=filter_strides
    )
    x = np.transpose(x, (1, 0, 2, 3))
    res = conv2d(x, filters, strides, padding, "NHWC", dilations)
    res = np.transpose(res, (1, 0, 2, 3))
    res = np.lib.stride_tricks.as_strided(
        res, shape=res.shape[1:], strides=res.strides[1:]
    )
    if data_format == "NCW":
        res = np.transpose(res, (0, 2, 1))
    return res


def conv1d_transpose(*_):
    raise Exception("Convolutions not yet implemented for numpy library")


def conv2d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    if isinstance(strides, int):
        strides = (strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0])

    if isinstance(dilations, int):
        dilations = (dilations, dilations)
    elif len(dilations) == 1:
        dilations = (dilations[0], dilations[0])

    # adding dilations
    if dilations[1] > 1:
        filters = np.insert(
            filters,
            [i for i in range(1, filters.shape[1])] * (dilations[1] - 1),
            values=0,
            axis=1,
        )
    if dilations[0] > 1:
        filters = np.insert(
            filters,
            [i for i in range(1, filters.shape[0])] * (dilations[0] - 1),
            values=0,
            axis=0,
        )

    filter_shape = filters.shape[0:2]
    filter_shape = list(filter_shape)

    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))

    x_shape = list(x.shape[1:3])
    if padding == "SAME":
        if x_shape[1] % strides[1] == 0:
            pad_w = max(filter_shape[1] - strides[1], 0)
        else:
            pad_w = max(filter_shape[1] - (x_shape[1] % strides[1]), 0)

        if x_shape[0] % strides[0] == 0:
            pad_h = max(filter_shape[0] - strides[0], 0)
        else:
            pad_h = max(filter_shape[0] - (x_shape[0] % strides[0]), 0)
        x = np.pad(
            x,
            [
                (0, 0),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (0, 0),
            ],
            "constant",
        )

    x_shape = x.shape
    input_dim = filters.shape[-2]
    output_dim = filters.shape[-1]
    new_h = (x_shape[1] - filter_shape[0]) // strides[0] + 1
    new_w = (x_shape[2] - filter_shape[1]) // strides[1] + 1
    new_shape = [x_shape[0], new_h, new_w] + filter_shape + [x_shape[-1]]
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
    # B x OH x OW x KH x KW x I x O
    sub_matrices_w_output_dim = np.tile(
        np.expand_dims(sub_matrices, -1), [1] * 6 + [output_dim]
    )
    # B x OH x OW x KH x KW x I x O
    mult = sub_matrices_w_output_dim * filters.reshape(
        [1] * 3 + filter_shape + [input_dim, output_dim]
    )
    # B x OH x OW x O
    res = np.sum(mult, (3, 4, 5))
    if data_format == "NCHW":
        return np.transpose(res, (0, 3, 1, 2))
    return res


def depthwise_conv2d(*_):
    raise Exception("Convolutions not yet implemented for numpy library")


def conv2d_transpose(*_):
    raise Exception("Convolutions not yet implemented for numpy library")


def conv3d(*_):
    raise Exception("Convolutions not yet implemented for numpy library")


def conv3d_transpose(*_):
    raise Exception("Convolutions not yet implemented for numpy library")
