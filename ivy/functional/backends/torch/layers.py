"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and
signature."""

from typing import Optional, Tuple, Union, Sequence

# global
import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from . import backend_version
from ivy.functional.ivy.layers import _get_embed_dim, _handle_padding, _deconv_length


@with_supported_dtypes(
    {"2.2 and below": ("float32", "float64", "complex")},
    backend_version,
)
def multi_head_attention(
    query: torch.Tensor,
    /,
    *,
    key: torch.Tensor = None,
    value: torch.Tensor = None,
    batch_first: bool = True,
    num_heads: Optional[int] = 8,
    scale: Optional[float] = None,
    attention_mask: torch.Tensor = None,
    in_proj_weights: torch.Tensor = None,
    q_proj_weights: torch.Tensor = None,
    k_proj_weights: torch.Tensor = None,
    v_proj_weights: torch.Tensor = None,
    out_proj_weights: torch.Tensor = None,
    in_proj_bias: torch.Tensor = None,
    out_proj_bias: torch.Tensor = None,
    is_causal: Optional[bool] = False,
    key_padding_mask: Optional[torch.Tensor] = None,
    bias_k: Optional[torch.Tensor] = None,
    bias_v: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
    add_zero_attn: bool = False,
    return_attention_weights: Optional[bool] = False,
    average_attention_weights: Optional[bool] = True,
    dropout: Optional[float] = 0.0,
    training: Optional[bool] = False,
    out: torch.Tensor = None,
) -> torch.Tensor:
    if key is None and value is None:
        key = value = query
    emb_dim = _get_embed_dim(
        in_proj_weights,
        q_proj_weights,
        k_proj_weights,
        v_proj_weights,
        query,
    )[1]
    num_dims = query.ndim
    if num_dims == 3 and batch_first:
        query, key, value = (torch.swapaxes(x, 0, 1) for x in [query, key, value])
    ret = torch.nn.functional.multi_head_attention_forward(
        query,
        key,
        value,
        emb_dim,
        num_heads,
        in_proj_weights,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout,
        out_proj_weights,
        out_proj_bias,
        training=training,
        key_padding_mask=key_padding_mask,
        need_weights=return_attention_weights,
        attn_mask=attention_mask,
        use_separate_proj_weight=not ivy.exists(in_proj_weights),
        q_proj_weight=q_proj_weights,
        k_proj_weight=k_proj_weights,
        v_proj_weight=v_proj_weights,
        static_k=static_k,
        static_v=static_v,
        average_attn_weights=average_attention_weights,
        is_causal=is_causal,
    )
    ret = list(ret) if isinstance(ret, tuple) else [ret]
    if num_dims == 3 and batch_first:
        ret[0] = ret[0].swapaxes(0, 1)
    if return_attention_weights:
        return tuple(ret)
    return ret[0]


multi_head_attention.partial_mixed_handler = (
    lambda *args, scale=None, out_proj_weights=None, is_causal=False, attention_mask=None, return_attention_weights=False, in_proj_weights=None, q_proj_weights=None, k_proj_weights=None, v_proj_weights=None, **kwargs: not ivy.exists(  # noqa: E501
        scale
    )
    and ivy.exists(out_proj_weights)
    and (not is_causal or ivy.exists(attention_mask))
    and (not is_causal or not return_attention_weights)
    and (
        ivy.exists(in_proj_weights)
        or all(ivy.exists(x) for x in [q_proj_weights, k_proj_weights, v_proj_weights])
    )
    and len(
        set(
            _get_embed_dim(
                in_proj_weights, q_proj_weights, k_proj_weights, v_proj_weights, args[0]
            )
        )
    )
    == 1
)


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    /,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


linear.partial_mixed_handler = lambda x, weight, **kwargs: weight.ndim == 2


def _x_dil_before_conv(x, dims, x_dilations):
    # adding dilation to input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    x_dilations_idxs = [i for i, x_dil in enumerate(x_dilations) if x_dil > 1]
    if x_dilations_idxs:
        for i in x_dilations_idxs:
            h = x.shape[2 + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = torch.eye(
                new_height,
                dtype=x.dtype,
                device=ivy.as_native_dev(ivy.default_device()),
            )[:: x_dilations[i]]
            x = torch.swapaxes(x, 2 + i, -1)
            x = torch.matmul(x, h)
            x = torch.swapaxes(x, -1, 2 + i)
    return x


def _pad_before_conv(
    x, filters, strides, padding, dims, dilations, filter_format="channel_last"
):
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    strides = [strides] * dims if isinstance(strides, int) else strides
    filter_shape = (
        filters.shape[2:] if filter_format == "channel_first" else filters.shape[:dims]
    )
    if isinstance(padding, str):
        # use torch's padding in conv if strides are all 1
        if len(strides) == strides.count(1):
            return x, padding.lower()
        filter_shape = [
            filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        pad_specific = [
            _handle_padding(x.shape[2 + i], strides[i], filter_shape[i], padding)
            for i in range(dims - 1, -1, -1)
        ]
        pad_list_top = [pad_specific[i] // 2 for i in range(dims)]
        pad_list_bot = [pad_specific[i] - pad_specific[i] // 2 for i in range(dims)]
        pad_list = [None] * len(pad_list_top) * 2
        pad_list[::2] = pad_list_top
        pad_list[1::2] = pad_list_bot
    else:
        if isinstance(padding, int):
            return x, padding
        # if symmetric padding is used, use torch's padding in conv function
        if all(pad[0] == pad[1] for pad in padding):
            return x, [pad[0] for pad in padding]
        pad_list = [item for sublist in padding for item in sublist[::-1]][::-1]
    return torch.nn.functional.pad(x, pad_list), 0


def _new_pad_before_conv(x, padding):
    if isinstance(padding, str):
        return x, padding.lower()
    elif isinstance(padding, int):
        return x, padding
    else:
        # if symmetric padding is used, use torch's padding in conv function
        if all(pad[0] == pad[1] for pad in padding):
            return x, [pad[0] for pad in padding]
        pad_list = [item for sublist in padding for item in sublist[::-1]][::-1]
        return torch.nn.functional.pad(x, pad_list), "valid"


def _tranpose_padding(
    x_shape, filter_shape, strides, padding, dims, dilations, output_shape, data_format
):
    if output_shape is not None and len(output_shape) > dims:
        if data_format[-1] == "C" or data_format == "channel_last":
            output_shape = output_shape[1:-1]
        elif data_format[1] == "C" or data_format == "channel_first":
            output_shape = output_shape[2:]
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    not_valid_pad = [False] * dims
    if isinstance(padding, str):
        if output_shape is None:
            output_shape = [
                _deconv_length(
                    x_shape[i], strides[i], filter_shape[i], padding, dilations[i]
                )
                for i in range(dims)
            ]
        if padding == "VALID":
            symmetric_padding = [0] * dims
        else:
            pad_specific = [
                _handle_padding(
                    output_shape[i],
                    strides[i],
                    filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1),
                    padding,
                )
                for i in range(dims)
            ]
            for i in range(dims):
                if pad_specific[i] % 2 != 0:
                    pad_specific[i] -= 1
                    not_valid_pad[i] = True
            symmetric_padding = [pad_specific[i] // 2 for i in range(dims)]
        out_shape = [
            (x_shape[i] - 1) * strides[i]
            - 2 * symmetric_padding[i]
            + dilations[i] * (filter_shape[i] - 1)
            + 1
            for i in range(dims)
        ]
        output_padding = [max(output_shape[i] - out_shape[i], 0) for i in range(dims)]
    else:
        if isinstance(padding, int):
            padding = [[padding, padding]] * dims
        symmetric_padding = [max(pad) for pad in padding]
        output_padding = [max(pad) - min(pad) for pad in padding]
    return not_valid_pad, symmetric_padding, output_padding


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if filter_format == "channel_last":
        filters = filters.permute(2, 1, 0)
    x = _x_dil_before_conv(x, 1, x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 1, dilations, "channel_first"
    )
    res = torch.nn.functional.conv1d(x, filters, bias, strides, padding, dilations)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv1d_v_1p9p0_and_above(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if filter_format == "channel_last":
        filters = filters.permute(2, 1, 0)
    x = _x_dil_before_conv(x, 1, x_dilations)
    if padding != "SAME" or all(
        s == 1 for s in ([strides] if isinstance(strides, int) else strides)
    ):
        x, padding = _new_pad_before_conv(x, padding)
    else:
        x, padding = _pad_before_conv(
            x, filters, strides, padding, 1, dilations, "channel_first"
        )
    res = torch.nn.functional.conv1d(x, filters, bias, strides, padding, dilations)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if filter_format == "channel_last":
        filters = filters.permute(2, 1, 0)
    not_valid_pad, symmetric_padding, output_padding = _tranpose_padding(
        x.shape[2:],
        filters.shape[2:],
        strides,
        padding,
        1,
        dilations,
        output_shape,
        data_format,
    )
    res = torch.nn.functional.conv_transpose1d(
        x,
        filters,
        bias,
        strides,
        symmetric_padding,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[:, :, 0:-1]
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if filter_format == "channel_last":
        filters = filters.permute(3, 2, 0, 1)
    x = _x_dil_before_conv(x, 2, x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 2, dilations, "channel_first"
    )
    res = torch.nn.functional.conv2d(x, filters, bias, strides, padding, dilations)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv2d_v_1p9p0_and_above(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if filter_format == "channel_last":
        filters = filters.permute(3, 2, 0, 1)
    x = _x_dil_before_conv(x, 2, x_dilations)
    if padding != "SAME" or all(
        s == 1 for s in ([strides] if isinstance(strides, int) else strides)
    ):
        x, padding = _new_pad_before_conv(x, padding)
    else:
        x, padding = _pad_before_conv(
            x, filters, strides, padding, 2, dilations, "channel_first"
        )
    res = torch.nn.functional.conv2d(x, filters, bias, strides, padding, dilations)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv2d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if filter_format == "channel_last":
        filters = filters.permute(3, 2, 0, 1)
    not_valid_pad, symmetric_padding, output_padding = _tranpose_padding(
        x.shape[2:],
        filters.shape[2:],
        strides,
        padding,
        2,
        dilations,
        output_shape,
        data_format,
    )
    res = torch.nn.functional.conv_transpose2d(
        x,
        filters,
        bias,
        strides,
        symmetric_padding,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[..., :-1, :]
    if not_valid_pad[1]:
        res = res[..., :-1]
    if data_format == "NHWC":
        res = res.permute(0, *range(2, 4), 1)
    return res


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def depthwise_conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    filters = ivy.squeeze(filters, axis=3).to_native() if filters.ndim == 4 else filters
    filters = torch.unsqueeze(filters, -1)
    dims_in = filters.shape[-2]
    filters = filters.permute(2, 3, 0, 1)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 2, dilations, "channel_first"
    )
    # noinspection PyArgumentEqualDefault
    res = torch.nn.functional.conv2d(
        x, filters, None, strides, padding, dilations, dims_in
    )
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")}, backend_version
)
# noinspection PyUnresolvedReferences
def conv3d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if filter_format == "channel_last":
        filters = filters.permute(4, 3, 0, 1, 2)
    x = _x_dil_before_conv(x, 3, x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 3, dilations, "channel_first"
    )
    res = torch.nn.functional.conv3d(x, filters, bias, strides, padding, dilations)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")}, backend_version
)
def conv3d_v_1p9p0_and_above(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if filter_format == "channel_last":
        filters = filters.permute(4, 3, 0, 1, 2)
    x = _x_dil_before_conv(x, 3, x_dilations)
    if padding != "SAME" or all(
        s == 1 for s in ([strides] if isinstance(strides, int) else strides)
    ):
        x, padding = _new_pad_before_conv(x, padding)
    else:
        x, padding = _pad_before_conv(
            x, filters, strides, padding, 3, dilations, "channel_first"
        )
    res = torch.nn.functional.conv3d(x, filters, bias, strides, padding, dilations)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv3d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if filter_format == "channel_last":
        filters = filters.permute(4, 3, 0, 1, 2)
    not_valid_pad, symmetric_padding, output_padding = _tranpose_padding(
        x.shape[2:],
        filters.shape[2:],
        strides,
        padding,
        3,
        dilations,
        output_shape,
        data_format,
    )
    res = torch.nn.functional.conv_transpose3d(
        x,
        filters,
        bias,
        strides,
        symmetric_padding,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[:, :, 0:-1, :, :]
    if not_valid_pad[1]:
        res = res[:, :, :, 0:-1, :]
    if not_valid_pad[2]:
        res = res[:, :, :, :, 0:-1]
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv_general_dilated(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    # permuting dims based on formats
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))
    if filter_format == "channel_last":
        filters = filters.permute(-1, -2, *range(dims))

    x = _x_dil_before_conv(x, dims, x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, dims, dilations, "channel_first"
    )

    if dims == 1:
        res = torch.nn.functional.conv1d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    elif dims == 2:
        res = torch.nn.functional.conv2d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    else:
        res = torch.nn.functional.conv3d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )

    if data_format == "channel_last":
        return res.permute(0, *range(2, dims + 2), 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv_general_dilated_v_1p9p0_and_above(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    # permuting dims based on formats
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))
    if filter_format == "channel_last":
        filters = filters.permute(-1, -2, *range(dims))

    x = _x_dil_before_conv(x, dims, x_dilations)
    if padding != "SAME" or all(
        s == 1 for s in ([strides] if isinstance(strides, int) else strides)
    ):
        x, padding = _new_pad_before_conv(x, padding)
    else:
        x, padding = _pad_before_conv(
            x, filters, strides, padding, dims, dilations, "channel_first"
        )

    if dims == 1:
        res = torch.nn.functional.conv1d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    elif dims == 2:
        res = torch.nn.functional.conv2d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    else:
        res = torch.nn.functional.conv3d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )

    if data_format == "channel_last":
        return res.permute(0, *range(2, dims + 2), 1)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv_general_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "channel_first",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))
    if filter_format == "channel_last":
        filters = filters.permute(dims + 1, dims, *range(dims))
    not_valid_pad, symmetric_padding, output_padding = _tranpose_padding(
        x.shape[2:],
        filters.shape[2:],
        strides,
        padding,
        dims,
        dilations,
        output_shape,
        data_format,
    )
    if dims == 1:
        res = torch.nn.functional.conv_transpose1d(
            x,
            filters,
            bias,
            strides,
            symmetric_padding,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[:, :, :-1]
    elif dims == 2:
        res = torch.nn.functional.conv_transpose2d(
            x,
            filters,
            bias,
            strides,
            symmetric_padding,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[..., :-1, :]
        if not_valid_pad[1]:
            res = res[..., :-1]
    else:
        res = torch.nn.functional.conv_transpose3d(
            x,
            filters,
            bias,
            strides,
            symmetric_padding,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[..., :-1, :, :]
        if not_valid_pad[1]:
            res = res[..., :, :-1, :]
        if not_valid_pad[2]:
            res = res[..., :, :, :-1]
    if data_format == "channel_last":
        res = res.permute(0, *range(2, dims + 2), 1)
    return res


def scaled_dot_product_attention_v_2p0p0_and_above(
    q,
    k,
    v,
    scale: float,
    /,
    *,
    mask=None,
    out=None,
):
    if isinstance(mask, torch.Tensor):
        mask = torch.where(mask == 0, -torch.inf, 0)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def lstm(
    input: torch.Tensor,
    initial_states: Tuple[torch.Tensor],
    all_weights: Tuple[torch.Tensor],
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool = False,
    batch_sizes: Sequence = None,
    weights_transposed: bool = False,
    has_ih_bias: bool = True,
    has_hh_bias: bool = True,
):
    if weights_transposed:
        # transpose the weights if they are in the wrong format
        all_weights = [
            torch.transpose(weight, 1, 0).contiguous() if weight.dim() == 2 else weight
            for weight in all_weights
        ]
    else:
        all_weights = list(all_weights)

    if (has_ih_bias and not has_hh_bias) or (has_hh_bias and not has_ih_bias):
        # insert zero biases into the weights where one set of biases is not
        # used, to avoid stride errors in lstm
        shapes = []
        for i in range(2, len(all_weights), 3):
            shapes.append(tuple(all_weights[i].shape))
        for i, shape in enumerate(shapes):
            idx = (i + 1) * 4 - (1 if has_ih_bias else 2)
            all_weights.insert(idx, torch.zeros(shape))
        has_ih_bias = True
        has_hh_bias = True

    if initial_states[0].dim() == 2:
        initial_states[0] = ivy.expand_dims(initial_states[0])
    if initial_states[1].dim() == 2:
        initial_states[1] = ivy.expand_dims(initial_states[1])

    ret = torch.lstm(
        input,
        initial_states,
        all_weights,
        has_ih_bias,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )

    return ret[0][:, -1], ret[0], (ret[1], ret[2])
