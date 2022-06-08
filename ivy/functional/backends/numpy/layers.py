"""Collection of Numpy network layers, wrapped to fit Ivy syntax and signature."""

# global
import numpy as np
from typing import Union, Tuple, Optional


def scaled_dot_product_attention(
    q: np.ndarray, 
    k: np.ndarray,
    v: np.ndarray, 
    scale: float, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:

    attention = np.matmul(q, np.transpose(k)) * scale

    if mask is not None:
        attention += (mask * -1e9)
    
    attention = np.exp(attention)/ np.sum(np.exp(attention), axis=-1, keepdims=True)
    output = np.matmul(attention, v)

    return output, attention

def conv1d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: int,
    padding: str,
    data_format: str = "NWC",
    dilations: int = 1,
) -> np.ndarray:
    x_shape = (1,) + x.shape
    filter_shape = (1,) + filters.shape
    x_strides = (x.strides[0],) + x.strides
    filter_strides = (filters.strides[0],) + filters.strides
    x = np.lib.stride_tricks.as_strided(x, shape=x_shape, strides=x_strides)
    filters = np.lib.stride_tricks.as_strided(
        filters, shape=filter_shape, strides=filter_strides
    )
    res = conv2d(x, filters, strides, padding, data_format, dilations)
    res = np.lib.stride_tricks.as_strided(
        res, shape=res.shape[1:], strides=res.strides[1:]
    )
    return res


def conv1d_transpose(*_):
    raise Exception("Convolutions not yet implemented for numpy library")


def conv2d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: int = 1,
) -> np.ndarray:
    filter_shape = filters.shape[0:2]
    filter_shape = list(filter_shape)
    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
    if padding == "SAME":
        x = np.pad(
            x, [[0, 0], [filter_shape[0] // 2] * 2, [filter_shape[1] // 2] * 2, [0, 0]]
        )
    x_shape = x.shape
    input_dim = filters.shape[-2]
    output_dim = filters.shape[-1]
    new_h = x_shape[1] - filter_shape[0] + 1
    new_w = x_shape[2] - filter_shape[1] + 1
    new_shape = [x_shape[0], new_h, new_w] + filter_shape + [x_shape[-1]]
    # ToDo: add non-unit stride support
    new_strides = x.strides[0:1] + x.strides[1:3] + x.strides[1:3] + x.strides[-1:]
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
