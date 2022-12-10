"""Collection of Jax network layers, wrapped to fit Ivy syntax and signature."""

# global
import jax.lax as jlax
import jax.numpy as jnp

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from typing import Union, Tuple, Optional, Sequence

# Extra #
# ------#


def _conv_transpose_padding(k, s, padding, dilation, diff=0):
    k = (k - 1) * dilation + 1
    if padding == "SAME":
        pad_len = k + s - 2
        pad_len -= diff
        if s > k - 1:
            pad_a = k - 1
        else:
            pad_a = int(jnp.ceil(pad_len / 2))
    else:
        pad_len = k + s - 2 + max(k - s, 0)
        pad_a = k - 1
    pad_b = pad_len - pad_a
    return pad_a, pad_b


def conv1d(
    x: JaxArray,
    filters: JaxArray,
    strides: int,
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = (strides,) if isinstance(strides, int) else strides
    dilations = (dilations,) if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x, filters, strides, padding, None, dilations, (data_format, "WIO", data_format)
    )


def conv1d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: int,
    padding: str,
    /,
    *,
    output_shape=None,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = (strides,) if isinstance(strides, int) else strides
    dilations = (dilations,) if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    if data_format == "NWC":
        x_shape = list(x.shape[1:2])
    else:
        x_shape = list(x.shape[2:])
    out_w = ivy.deconv_length(
        x_shape[0], strides[0], filters.shape[0], padding, dilations[0]
    )

    if output_shape is None:
        output_shape = [x_shape[0], out_w, filters.shape[-1]]
    elif len(output_shape) == 1:
        output_shape = [x_shape[0], output_shape[0], filters.shape[-1]]
    diff_w = -(output_shape[1] - out_w)
    pad_w_before, pad_w_after = _conv_transpose_padding(
        filters.shape[0], strides[0], padding, dilations[0], diff_w
    )
    return jlax.conv_transpose(
        x,
        filters,
        strides,
        [(pad_w_before, pad_w_after)],
        dilations,
        (data_format, "WIO", data_format),
        True,
    )


def conv2d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    dilations: int = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        None,
        dilations,
        (data_format, "HWIO", data_format),
    )


def depthwise_conv2d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    dilations: int = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    cn = filters.shape[-1]
    filters = jnp.expand_dims(filters, -2)
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        None,
        dilations,
        (data_format, "HWIO", data_format),
        feature_group_count=cn,
    )


def conv2d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape=None,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    if data_format == "NHWC":
        x_shape = list(x.shape[1:3])
    else:
        x_shape = list(x.shape[2:])
    out_h = ivy.deconv_length(
        x_shape[0], strides[0], filters.shape[0], padding, dilations[0]
    )
    out_w = ivy.deconv_length(
        x_shape[1], strides[1], filters.shape[1], padding, dilations[1]
    )
    if output_shape is None:
        output_shape = [x.shape[0], out_h, out_w, filters.shape[-2]]
    elif len(output_shape) == 2:
        output_shape = [x.shape[0], output_shape[0], output_shape[1], filters.shape[-2]]
    diff_h = -(output_shape[1] - out_h)
    diff_w = -(output_shape[2] - out_w)
    pad_h_before, pad_h_after = _conv_transpose_padding(
        filters.shape[0], strides[0], padding, dilations[0], diff_h
    )
    pad_w_before, pad_w_after = _conv_transpose_padding(
        filters.shape[1], strides[1], padding, dilations[1], diff_w
    )
    return jlax.conv_transpose(
        x,
        filters,
        strides,
        [(pad_h_before, pad_h_after), (pad_w_before, pad_w_after)],
        dilations,
        (data_format, "HWIO", data_format),
        True,
    )


def conv3d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        None,
        dilations,
        (data_format, "DHWIO", data_format),
    )


def conv3d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape=None,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    data_format: str = "NDHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    if data_format == "NDHWC":
        x_shape = list(x.shape[1:4])
    else:
        x_shape = list(x.shape[2:])
    out_d = ivy.deconv_length(
        x_shape[0], strides[0], filters.shape[0], padding, dilations[0]
    )
    out_h = ivy.deconv_length(
        x_shape[1], strides[1], filters.shape[1], padding, dilations[1]
    )
    out_w = ivy.deconv_length(
        x_shape[2], strides[2], filters.shape[2], padding, dilations[2]
    )
    if output_shape is None:
        output_shape = [x.shape[0], out_d, out_h, out_w, filters.shape[-2]]
    elif len(output_shape) == 3:
        output_shape = [
            x.shape[0],
            output_shape[0],
            output_shape[1],
            output_shape[2],
            filters.shape[-2],
        ]
    diff_d = -(output_shape[1] - out_d)
    diff_h = -(output_shape[2] - out_h)
    diff_w = -(output_shape[3] - out_w)
    pad_d_before, pad_d_after = _conv_transpose_padding(
        filters.shape[0], strides[0], padding, dilations[0], diff_d
    )
    pad_h_before, pad_h_after = _conv_transpose_padding(
        filters.shape[1], strides[1], padding, dilations[1], diff_h
    )
    pad_w_before, pad_w_after = _conv_transpose_padding(
        filters.shape[2], strides[2], padding, dilations[2], diff_w
    )
    return jlax.conv_transpose(
        x,
        filters,
        strides,
        [
            (pad_d_before, pad_d_after),
            (pad_h_before, pad_h_after),
            (pad_w_before, pad_w_after),
        ],
        dilations,
        (data_format, "DHWIO", data_format),
        True,
    )


def _get_filter_dataformat(dims: int = 2):
    if dims == 1:
        return "WIO"
    if dims == 2:
        return "HWIO"
    elif dims == 3:
        return "DHWIO"


def conv_general_dilated(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    out: Optional[JaxArray] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    filter_df = _get_filter_dataformat(dims)
    if not len(x_dilations) == x_dilations.count(1):
        new_pad = [0] * dims
        if data_format == "channel_last":
            x_shape = list(x.shape[1 : dims + 1])
        else:
            x_shape = list(x.shape[2:])
        x_shape = [
            x_shape[i] + (x_shape[i] - 1) * (x_dilations[i] - 1) for i in range(dims)
        ]
        filter_shape = list(filters.shape[:dims])
        filter_shape = [
            filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        for i in range(dims):
            new_pad[i] = ivy.handle_padding(
                x_shape[i], strides[i], filter_shape[i], padding
            )
        padding = [(new_pad[i] // 2, new_pad[i] - new_pad[i] // 2) for i in range(dims)]
    df = ivy.get_x_data_format(dims, data_format)
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        x_dilations,
        dilations,
        (df, filter_df, df),
        feature_group_count,
    )


def conv_general_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    output_shape=None,
    data_format: str = "channel_last",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    out: Optional[JaxArray] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    df = ivy.get_x_data_format(dims, "channel_last")
    filter_df = _get_filter_dataformat(dims)
    if data_format == "channel_first":
        x = jnp.transpose(x, (0, *range(2, dims + 2), 1))
    x_shape = list(x.shape[1 : dims + 1])
    out_shape = [
        ivy.deconv_length(
            x_shape[i], strides[i], filters.shape[i], padding, dilations[i]
        )
        for i in range(dims)
    ]
    if output_shape is None:
        output_shape = [x.shape[0]] + out_shape + [filters.shape[-2]]
    elif len(output_shape) == dims:
        output_shape = [x.shape[0], *output_shape, filters.shape[-2]]

    diff = [-(output_shape[i + 1] - out_shape[i]) for i in range(dims)]
    pad = [0] * dims
    for i in range(dims):
        pad[i] = _conv_transpose_padding(
            filters.shape[i], strides[i], padding, dilations[i], diff[i]
        )
    res = jnp.concatenate(
        [
            jlax.conv_transpose(
                x[..., j : j + filters.shape[-1] // feature_group_count],
                filters[..., j : j + filters.shape[-1] // feature_group_count],
                strides,
                pad,
                dilations,
                (df, filter_df, df),
                True,
            )
            for j in range(
                0, filters.shape[-1], filters.shape[-1] // feature_group_count
            )
        ],
        axis=-1,
    )
    if data_format == "channel_first":
        return jnp.transpose(res, (0, dims + 1, *range(1, dims + 1)))
    return res
