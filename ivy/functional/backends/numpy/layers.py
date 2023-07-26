"""Collection of Numpy network layers, wrapped to fit Ivy syntax and signature."""

# global
import numpy as np
from typing import Union, Tuple, Optional, Sequence


# local
import ivy
from ivy.functional.ivy.layers import (
    _handle_padding,
    _deconv_length,
    _get_x_data_format,
)


def _add_dilations(x, dilations, axis, values=0):
    return np.insert(
        x,
        [i for i in range(1, x.shape[axis])] * (dilations - 1),
        values=values,
        axis=axis,
    )


def _dilate_pad_conv(x, filters, strides, padding, dims, dilations):
    for j in range(dims):
        if dilations[j] > 1:
            filters = _add_dilations(filters, dilations[j], axis=j)
    if isinstance(padding, str):
        pad_specific = [
            _handle_padding(x.shape[1 + i], strides[i], filters.shape[i], padding)
            for i in range(dims)
        ]
        pad_list = [
            (pad_specific[i] // 2, pad_specific[i] - pad_specific[i] // 2)
            for i in range(dims)
        ]
    elif isinstance(padding, int):
        pad_list = [(padding, padding)] * dims
    else:
        pad_list = [(_p, _p) if isinstance(_p, int) else _p for _p in padding]

    pad_width = [(0, 0), *pad_list, (0, 0)]

    x = np.pad(
        x,
        pad_width=pad_width,
        mode="constant",
    )
    return x, filters


def _dilate_pad_conv_tranpose(
    x, filters, strides, padding, dims, dilations, output_shape
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    if output_shape is None:
        new_shape = [
            _deconv_length(
                x.shape[i + 1], strides[i], filters.shape[i], padding, dilations[i]
            )
            for i in range(dims)
        ]
        output_shape = [x.shape[0], *new_shape, filters.shape[-1]]
    elif len(output_shape) == dims:
        output_shape = [x.shape[0]] + list(output_shape) + [filters.shape[-1]]
    for i in reversed(range(dims)):
        if strides[i] > 1:
            x = _add_dilations(x, strides[i], axis=i + 1)
        if dilations[i] > 1:
            filters = _add_dilations(filters, dilations[i], axis=i)
    pad_specific = [
        _handle_padding(output_shape[i + 1], strides[i], filters.shape[i], padding)
        for i in range(dims)
    ]
    extra_pad = [
        max(
            0,
            output_shape[i + 1]
            - (x.shape[i + 1] + filters.shape[i] - 1 - pad_specific[i]),
        )
        for i in range(dims)
    ]
    pad_top = [filters.shape[i] - 1 - (pad_specific[i] // 2) for i in range(dims)]
    pad_bot = [
        filters.shape[i] - 1 - (pad_specific[i] - pad_specific[i] // 2)
        for i in range(dims)
    ]
    pad_list = [(pad_top[i], pad_bot[i] + extra_pad[i]) for i in range(dims)]
    x = np.pad(
        x,
        [
            (0, 0),
            *pad_list,
            (0, 0),
        ],
        "constant",
    )
    return x, filters


def conv1d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int]] = 1,
    padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    /,
    *,
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    strides = [strides] if isinstance(strides, int) else strides
    dilations = [dilations] if isinstance(dilations, int) else dilations
    if data_format == "NCW":
        x = np.transpose(x, (0, 2, 1))

    x, filters = _dilate_pad_conv(x, filters, strides, padding, 1, dilations)

    x_shape = x.shape
    filter_shape = list(filters.shape[0:1])
    input_dim = filters.shape[-2]
    output_dim = filters.shape[-1]
    new_w = (x_shape[1] - filter_shape[0]) // strides[0] + 1
    new_shape = [x_shape[0], new_w] + filter_shape + [x_shape[-1]]
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
    # B x OW x KW x I x O
    sub_matrices_w_output_dim = np.tile(
        np.expand_dims(sub_matrices, -1), [1] * 4 + [output_dim]
    )
    # B x OW x KW x I x O
    mult = sub_matrices_w_output_dim * filters.reshape(
        [1] * 2 + filter_shape + [input_dim, output_dim]
    )
    # B x OW x O
    res = np.sum(mult, (2, 3))

    if data_format == "NCW":
        res = np.transpose(res, (0, 2, 1))
    return res


def conv1d_transpose(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int]] = 1,
    padding: str = "VALID",
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if data_format == "NCW":
        x = np.transpose(x, (0, 2, 1))
    x, filters = _dilate_pad_conv_tranpose(
        x, filters, strides, padding, 1, dilations, output_shape
    )
    x = np.flip(x, (1,))
    res = np.flip(
        conv1d(x, filters, 1, "VALID", data_format="NWC", dilations=1),
        (1,),
    )
    if data_format == "NCW":
        res = np.transpose(res, (0, 2, 1))
    return res


def conv2d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))

    x, filters = _dilate_pad_conv(x, filters, strides, padding, 2, dilations)

    x_shape = x.shape
    filter_shape = list(filters.shape[0:2])
    input_dim = filters.shape[-2]
    output_dim = filters.shape[-1]
    new_h = (x_shape[1] - filter_shape[0]) // strides[0] + 1
    new_w = (x_shape[2] - filter_shape[1]) // strides[1] + 1
    new_shape = [x_shape[0], new_h, new_w] + filter_shape + [x_shape[-1]]
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


def conv2d_transpose(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = "VALID",
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[np.ndarray] = None,
):
    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
    x, filters = _dilate_pad_conv_tranpose(
        x, filters, strides, padding, 2, dilations, output_shape
    )
    x = np.flip(x, (1, 2))
    res = np.flip(
        conv2d(x, filters, 1, "VALID", data_format="NHWC", dilations=1),
        (1, 2),
    )
    if data_format == "NCHW":
        res = np.transpose(res, (0, 3, 1, 2))
    return res


def depthwise_conv2d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[np.ndarray] = None,
):
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if isinstance(padding, int):
        padding = [(padding, padding)] * 2
    if data_format == "NHWC":
        x = np.transpose(x, (3, 0, 1, 2))
    else:
        x = np.transpose(x, (1, 0, 2, 3))
    filters = np.squeeze(filters, 3) if filters.ndim == 4 else filters
    filters = np.transpose(filters, (2, 0, 1))
    filters = np.expand_dims(filters, (-1, -2))
    filter_h = filters.shape[1] + (filters.shape[1] - 1) * (dilations[0] - 1)
    filter_w = filters.shape[2] + (filters.shape[2] - 1) * (dilations[1] - 1)
    if isinstance(padding, str):
        if padding == "VALID":
            out_height = np.ceil(float(x.shape[2] - filter_h + 1) / float(strides[0]))
            out_width = np.ceil(float(x.shape[3] - filter_w + 1) / float(strides[1]))
        else:
            out_height = np.ceil(float(x.shape[2]) / float(strides[0]))
            out_width = np.ceil(float(x.shape[3]) / float(strides[1]))
    else:
        out_height = np.ceil(
            float(x.shape[2] - filter_h + padding[0][0] + padding[0][1] + 1)
            / float(strides[0])
        )
        out_width = np.ceil(
            float(x.shape[3] - filter_w + padding[1][0] + padding[1][1] + 1)
            / float(strides[1])
        )
    if data_format == "NHWC":
        outputs = np.empty([x.shape[1], int(out_height), int(out_width), 0], x.dtype)
    else:
        outputs = np.empty([x.shape[1], 0, int(out_height), int(out_width)], x.dtype)
    x = np.expand_dims(x, -1)
    for i in range(x.shape[0]):
        output = conv2d(
            x[i], filters[i], strides, padding, data_format="NHWC", dilations=dilations
        )
        if data_format == "NHWC":
            outputs = np.append(outputs, output, axis=-1)
        else:
            outputs = np.append(outputs, np.transpose(output, (0, 3, 1, 2)), axis=1)
    return outputs


def conv3d(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    /,
    *,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    if data_format == "NCDHW":
        x = np.transpose(x, (0, 2, 3, 4, 1))

    x, filters = _dilate_pad_conv(x, filters, strides, padding, 3, dilations)

    x_shape = x.shape
    filter_shape = list(filters.shape[0:3])
    input_dim = filters.shape[-2]
    output_dim = filters.shape[-1]
    new_d = (x_shape[1] - filter_shape[0]) // strides[0] + 1
    new_h = (x_shape[2] - filter_shape[1]) // strides[1] + 1
    new_w = (x_shape[3] - filter_shape[2]) // strides[2] + 1
    new_shape = [x_shape[0], new_d, new_h, new_w] + filter_shape + [x_shape[-1]]
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
    # B x OD X OH x OW x KD X KH x KW x I
    sub_matrices = np.lib.stride_tricks.as_strided(
        x, new_shape, new_strides, writeable=False
    )
    # B x OD X OH x OW x KD X KH x KW x I x O
    sub_matrices_w_output_dim = np.tile(
        np.expand_dims(sub_matrices, -1), [1] * 8 + [output_dim]
    )
    # B x OD X OH x OW x KD X KH x KW x I x O
    mult = sub_matrices_w_output_dim * filters.reshape(
        [1] * 4 + filter_shape + [input_dim, output_dim]
    )
    # B x OD X OH x OW x O
    res = np.sum(mult, (4, 5, 6, 7))

    if data_format == "NCDHW":
        return np.transpose(res, (0, 4, 1, 2, 3))
    return res


def conv3d_transpose(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int, int, int]] = 1,
    padding: str = "VALID",
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    out: Optional[np.ndarray] = None,
):
    if data_format == "NCDHW":
        x = np.transpose(x, (0, 2, 3, 4, 1))
    x, filters = _dilate_pad_conv_tranpose(
        x, filters, strides, padding, 3, dilations, output_shape
    )
    x = np.flip(x, (1, 2, 3))
    res = np.flip(
        conv3d(x, filters, 1, "VALID", data_format="NDHWC", dilations=1),
        (1, 2, 3),
    )
    if data_format == "NCDHW":
        res = np.transpose(res, (0, 4, 1, 2, 3))
    return res


def conv_general_dilated(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[np.ndarray] = None,
    out: np.ndarray = None,
) -> np.ndarray:
    # permuting dims based on formats
    if data_format == "channel_first":
        x = np.transpose(x, (0, *range(2, dims + 2), 1))
    if filter_format == "channel_first":
        filters = np.transpose(filters, (*range(2, dims + 2), 1, 0))

    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations

    for j in range(dims):
        if x_dilations[j] > 1:
            x = _add_dilations(x, x_dilations[j], axis=j + 1)
    x, filters = _dilate_pad_conv(x, filters, strides, padding, dims, dilations)

    x_shape = x.shape
    filter_shape = list(filters.shape[0:dims])
    input_dim = filters.shape[-2]
    output_dim = filters.shape[-1]
    new_shape = [
        (x_shape[i + 1] - filter_shape[i]) // strides[i] + 1 for i in range(dims)
    ]
    res = []
    new_shape = [x_shape[0], *new_shape] + filter_shape + [input_dim]
    for i, j in zip(
        range(0, x.shape[-1], input_dim),
        range(0, output_dim, output_dim // feature_group_count),
    ):
        sliced_x = x[..., i : i + input_dim]
        sliced_filters = filters[..., j : j + output_dim // feature_group_count]
        normal_strides = [sliced_x.strides[i] for i in range(1, dims + 2)]
        changed_strides = [
            sliced_x.strides[i] * strides[i - 1] for i in range(1, dims + 1)
        ]
        new_strides = (x.strides[0], *changed_strides, *normal_strides)
        # B x OH x OW x KH x KW x I
        sub_matrices = np.lib.stride_tricks.as_strided(
            sliced_x, new_shape, new_strides, writeable=False
        )

        # B x OH x OW x KH x KW x I x O
        sub_matrices_w_output_dim = np.tile(
            np.expand_dims(sub_matrices, -1),
            [1] * (dims * 2 + 2) + [output_dim // feature_group_count],
        )
        # B x OH x OW x KH x KW x I x O
        mult = sub_matrices_w_output_dim * sliced_filters.reshape(
            [1] * (dims + 1)
            + filter_shape
            + [input_dim, output_dim // feature_group_count]
        )

        # B x OH x OW x O
        res.append(np.sum(mult, tuple([i for i in range(dims + 1, dims * 2 + 2)])))
    res = np.concatenate(res, axis=-1)
    res = np.add(res, bias) if bias is not None else res

    if data_format == "channel_first":
        return np.transpose(res, (0, dims + 1, *range(1, dims + 1)))
    return res


def conv_general_transpose(
    x: np.ndarray,
    filters: np.ndarray,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    padding: str = "VALID",
    /,
    *,
    dims: int = 2,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "channel_last",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if data_format == "channel_first":
        x = np.transpose(x, (0, *range(2, dims + 2), 1))

    x, filters = _dilate_pad_conv_tranpose(
        x, filters, strides, padding, dims, dilations, output_shape
    )

    x = np.flip(x, (*range(1, dims + 1),))
    res = np.concatenate(
        [
            np.flip(
                conv_general_dilated(
                    x[..., j : j + filters.shape[-2] // feature_group_count],
                    filters[..., j : j + filters.shape[-2] // feature_group_count, :],
                    1,
                    "VALID",
                    dims=dims,
                    data_format=_get_x_data_format(dims, "channel_last"),
                    dilations=1,
                ),
                (*range(1, dims + 1),),
            )
            for j in range(
                0, filters.shape[-2], filters.shape[-2] // feature_group_count
            )
        ],
        axis=-1,
    )
    res = np.add(res, bias) if bias is not None else res

    if data_format == "channel_first":
        return np.transpose(res, (0, dims + 1, *range(1, dims + 1)))
    return res
