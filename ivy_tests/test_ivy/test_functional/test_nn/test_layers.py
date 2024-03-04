"""Collection of tests for unified neural network layers."""

# global
from hypothesis import strategies as st, assume
import ivy
import numpy as np


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
from ivy.functional.ivy.layers import _deconv_length, _pack_padded_sequence


# --- Helpers --- #
# --------------- #


def _assume_tf_dilation_gt_1(backend_fw, on_device, dilations):
    if backend_fw == "tensorflow":
        assume(
            not (
                on_device == "cpu" and (dilations > 1)
                if isinstance(dilations, int)
                else any(d > 1 for d in dilations)
            )
        )


# Dropout #
# --------#


@st.composite
def _dropout_helper(draw):
    mixed_fn_compos = draw(st.booleans())
    is_torch_backend = ivy.current_backend_str() == "torch"
    shape = draw(helpers.get_shape(min_num_dims=1))
    dtype = draw(
        helpers.get_dtypes("float", full=False, mixed_fn_compos=mixed_fn_compos)
    )
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "float", mixed_fn_compos=mixed_fn_compos
            ),
            shape=shape,
        )
    )
    noise_shape = list(shape)
    if draw(st.booleans()):
        noise_shape = None
    else:
        for i, _ in enumerate(noise_shape):
            if draw(st.booleans()):
                noise_shape[i] = 1
            elif draw(st.booleans()):
                noise_shape[i] = None
    seed = draw(helpers.ints(min_value=0, max_value=100))
    prob = draw(helpers.floats(min_value=0, max_value=0.9))
    scale = draw(st.booleans())
    training = draw(st.booleans())

    if is_torch_backend and not mixed_fn_compos:
        noise_shape = None
        seed = None
    return dtype_and_x, noise_shape, seed, dtype, prob, scale, training


@st.composite
def _general_transpose_helper(draw):
    dims = draw(st.integers(1, 3))
    padding = st.sampled_from(["SAME", "VALID"]) if dims != 2 else None
    x_f_d_df = draw(
        _x_and_filters(
            dim=dims,
            general=True,
            transpose=True,
            bias=True,
            padding=padding,
        )
    )
    return dims, x_f_d_df


@st.composite
def _lstm_helper(draw):
    dtype = draw(helpers.get_dtypes("float", full=False))

    has_ih_bias = draw(st.booleans())
    has_hh_bias = draw(st.booleans())
    weights_transposed = draw(st.booleans())
    bidirectional = draw(st.booleans())
    dropout = draw(st.floats(min_value=0, max_value=0.99))
    train = draw(st.booleans()) and not dropout
    packed = draw(st.booleans())

    batch_first = draw(st.booleans()) and not packed
    num_batches = draw(st.integers(min_value=1, max_value=5))
    num_layers = draw(st.integers(min_value=1, max_value=3))
    num_directions = 2 if bidirectional else 1
    seq_size = draw(st.integers(min_value=1, max_value=5))
    in_size = draw(st.integers(min_value=1, max_value=3))
    hidden_size = draw(st.integers(min_value=1, max_value=3))

    input = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(
                (num_batches, seq_size, in_size)
                if batch_first
                else (seq_size, num_batches, in_size)
            ),
            min_value=0,
            max_value=1,
        )
    )

    init_h = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(num_directions * num_layers, num_batches, hidden_size),
            min_value=0,
            max_value=1,
        )
    )
    init_c = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(num_directions * num_layers, num_batches, hidden_size),
            min_value=0,
            max_value=1,
        )
    )

    all_weights = []
    for k in range(num_layers):
        for _ in range(num_directions):
            weight_ih = draw(
                helpers.array_values(
                    dtype=dtype[0],
                    shape=(
                        (4 * hidden_size, in_size)
                        if k == 0
                        else (4 * hidden_size, num_directions * hidden_size)
                    ),
                    min_value=0,
                    max_value=1,
                )
            )
            weight_hh = draw(
                helpers.array_values(
                    dtype=dtype[0],
                    shape=(4 * hidden_size, hidden_size),
                    min_value=0,
                    max_value=1,
                )
            )
            all_weights += [weight_ih, weight_hh]
            if has_ih_bias:
                bias_ih = draw(
                    helpers.array_values(
                        dtype=dtype[0],
                        shape=(4 * hidden_size,),
                        min_value=0,
                        max_value=1,
                    )
                )
                all_weights.append(bias_ih)
            if has_hh_bias:
                bias_hh = draw(
                    helpers.array_values(
                        dtype=dtype[0],
                        shape=(4 * hidden_size,),
                        min_value=0,
                        max_value=1,
                    )
                )
                all_weights.append(bias_hh)

    if weights_transposed:
        all_weights = [
            ivy.swapaxes(w, 0, 1) if w.dims() == 2 else w for w in all_weights
        ]

    if packed:
        batch_sizes = [seq_size]
        batch_sizes += draw(
            st.lists(
                st.integers(min_value=1, max_value=seq_size),
                min_size=num_batches - 1,
                max_size=num_batches - 1,
            )
        )
        batch_sizes = np.array(draw(st.permutations(batch_sizes)))
        input, batch_sizes = (
            ivy.to_numpy(p) for p in _pack_padded_sequence(input, batch_sizes)
        )
    else:
        batch_sizes = None

    initial_states = init_h, init_c
    all_weights = tuple(all_weights)
    if batch_sizes is not None:
        dtypes = dtype + ["int64"]
        kwargs = {
            "input": input,
            "batch_sizes": batch_sizes,
            "initial_states": initial_states,
            "all_weights": all_weights,
            "num_layers": num_layers,
            "dropout": dropout,
            "train": train,
            "bidirectional": bidirectional,
            "weights_transposed": weights_transposed,
            "has_ih_bias": has_ih_bias,
            "has_hh_bias": has_hh_bias,
        }
    else:
        dtypes = dtype
        kwargs = {
            "input": input,
            "initial_states": initial_states,
            "all_weights": all_weights,
            "num_layers": num_layers,
            "dropout": dropout,
            "train": train,
            "bidirectional": bidirectional,
            "batch_first": batch_first,
            "weights_transposed": weights_transposed,
            "has_ih_bias": has_ih_bias,
            "has_hh_bias": has_hh_bias,
        }
    return dtypes, kwargs


@st.composite
def _mha_helper(draw, same_pre_embed_dim=False, batch_second=False):
    _qkv_same_dim = draw(st.booleans())
    _self_attention = draw(st.booleans())
    _same_pre_embed_dim = _self_attention or same_pre_embed_dim or draw(st.booleans())
    batch_first = draw(st.booleans()) and not batch_second
    num_heads = draw(helpers.ints(min_value=1, max_value=3))
    _embed_dim = draw(helpers.ints(min_value=4, max_value=16)) * num_heads
    _batch_dim = draw(st.sampled_from([(), (1,)]))
    _num_batches = _batch_dim[0] if len(_batch_dim) else 1
    dtype = draw(helpers.get_dtypes("valid", full=False))
    _num_queries = draw(helpers.ints(min_value=2, max_value=8))
    _num_keys = draw(helpers.ints(min_value=2, max_value=8))
    in_proj_weights = None
    q_proj_weights = None
    k_proj_weights = None
    v_proj_weights = None

    if _qkv_same_dim:
        if _same_pre_embed_dim:
            _pre_embed_dim = _embed_dim
        else:
            _pre_embed_dim = draw(helpers.ints(min_value=4, max_value=16))
        q = draw(
            helpers.array_values(
                shape=(*_batch_dim, _num_queries, _pre_embed_dim),
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            )
        )
        k = draw(
            helpers.array_values(
                shape=(*_batch_dim, _num_keys, _pre_embed_dim),
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            )
            if not _self_attention
            else st.none()
        )
        v = draw(
            helpers.array_values(
                shape=(*_batch_dim, _num_keys, _pre_embed_dim),
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            )
            if not _self_attention
            else st.none()
        )
        in_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(3 * _embed_dim, _pre_embed_dim),
                min_value=-10,
                max_value=10,
            )
            if not _same_pre_embed_dim or draw(st.booleans())
            else st.none()
        )
    else:
        if not same_pre_embed_dim:
            _q_dim = draw(helpers.ints(min_value=2, max_value=8))
        else:
            _q_dim = _embed_dim
        _k_dim = draw(helpers.ints(min_value=2, max_value=8))
        _v_dim = draw(helpers.ints(min_value=2, max_value=8))
        q = draw(
            helpers.array_values(
                shape=(*_batch_dim, _num_queries, _q_dim),
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            )
        )
        k = draw(
            helpers.array_values(
                shape=(*_batch_dim, _num_keys, _k_dim),
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            )
        )
        v = draw(
            helpers.array_values(
                shape=(*_batch_dim, _num_keys, _v_dim),
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            )
        )
        q_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_embed_dim, _q_dim),
                min_value=-5,
                max_value=5,
            )
        )
        k_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_embed_dim, _k_dim),
                min_value=-5,
                max_value=5,
            )
        )
        v_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_embed_dim, _v_dim),
                min_value=-5,
                max_value=5,
            )
        )
    in_proj_bias = draw(
        st.one_of(
            helpers.array_values(
                dtype=dtype[0],
                shape=(3 * _embed_dim,),
                min_value=-10,
                max_value=10,
            ),
            st.none(),
        )
    )

    _out_dim = draw(helpers.ints(min_value=4, max_value=16))
    out_proj_weights = draw(
        st.one_of(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_out_dim, _embed_dim),
                min_value=-5,
                max_value=5,
            ),
            st.none(),
        )
    )
    out_proj_bias = draw(
        st.one_of(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_out_dim,),
                min_value=-10,
                max_value=10,
            ),
            st.none(),
        )
    )

    if _self_attention and _qkv_same_dim:
        _num_keys = _num_queries
    _static_shape = (_num_batches * num_heads, _num_keys, int(_embed_dim // num_heads))
    static_k = draw(
        st.one_of(
            helpers.array_values(
                shape=_static_shape,
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            ),
            st.none(),
        )
    )
    static_v = draw(
        st.one_of(
            helpers.array_values(
                shape=_static_shape,
                dtype=dtype[0],
                max_value=1000,
                min_value=-1000,
                abs_smallest_val=1e-06,
            ),
            st.none(),
        )
    )

    _mask_shape = (_num_queries, _num_keys)
    if len(_batch_dim) and draw(st.booleans()):
        _mask_shape = (_num_batches * num_heads, *_mask_shape)
    attention_mask = draw(
        st.one_of(
            helpers.array_values(
                dtype=draw(st.sampled_from(["bool", dtype[0]])),
                allow_inf=True,
                shape=_mask_shape,
            ),
            st.none(),
        )
    )

    key_padding_mask = draw(
        st.one_of(
            helpers.array_values(
                dtype="bool",
                shape=(*_batch_dim, _num_keys),
            ),
            st.none(),
        )
    )

    _extra_bias = (
        (not _qkv_same_dim or _pre_embed_dim == _embed_dim)
        and static_k is None
        and static_v is None
        and draw(st.booleans())
    )
    bias_k = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(_embed_dim,), min_value=-10, max_value=10
        )
        if _extra_bias
        else st.none()
    )
    bias_v = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(_embed_dim,), min_value=-10, max_value=10
        )
        if _extra_bias
        else st.none()
    )

    scale = draw(st.one_of(st.floats(min_value=0.001), st.none()))
    add_zero_attn = draw(st.booleans())
    dropout = draw(st.floats(min_value=0, max_value=0.99))
    training = draw(st.booleans())
    is_causal = draw(st.booleans())
    return_attention_weights = draw(st.booleans())
    average_attention_weights = draw(st.booleans())

    if len(q.shape) == 3 and not batch_first:
        q, k, v = (np.swapaxes(x, 0, 1) if x is not None else x for x in [q, k, v])

    ret = (
        q,
        k,
        v,
        num_heads,
        attention_mask,
        in_proj_weights,
        q_proj_weights,
        k_proj_weights,
        v_proj_weights,
        out_proj_weights,
        in_proj_bias,
        out_proj_bias,
        key_padding_mask,
        bias_k,
        bias_v,
        static_k,
        static_v,
        scale,
        add_zero_attn,
        dropout,
        training,
        is_causal,
        return_attention_weights,
        average_attention_weights,
        batch_first,
    )
    ret_dtypes = [str(r.dtype) for r in ret if ivy.is_array(r)]
    return ret_dtypes, *ret


@st.composite
def _nms_helper(draw):
    img_width = draw(st.integers(250, 1250))
    img_height = draw(st.integers(250, 1250))
    num_boxes = draw(st.integers(5, 50))
    bbox = {}
    for _ in range(num_boxes):
        x1 = draw(st.integers(0, img_width - 20))
        w = draw(st.integers(5, img_width - x1))
        y1 = draw(st.integers(0, img_height - 20))
        h = draw(st.integers(5, img_height - y1))

        bbox[(x1, y1, x1 + w, y1 + h)] = draw(st.floats(0.2, 1))

    iou_threshold = draw(st.floats(0.2, 1))
    max_output_size = draw(st.integers(1, num_boxes))
    score_threshold = draw(st.floats(0, 1))
    return (
        np.array(list(bbox.keys()), dtype=np.float32),
        np.array(list(bbox.values()), dtype=np.float32),
        iou_threshold,
        max_output_size,
        score_threshold,
    )


# Convolutions #
# -------------#


def _output_shape(dims, dilation, stride, padding, x_shape, filter_shape):
    if isinstance(padding, str):
        return [
            _deconv_length(
                x_shape[i],
                stride[i],
                filter_shape[i],
                padding,
                dilation[i],
            )
            for i in range(dims)
        ]
    else:
        if isinstance(padding, int):
            padding = [[padding, padding]] * dims
        return [
            (x_shape[i] - 1) * stride[i]
            - padding[i][0]
            - padding[i][1]
            + dilation[i] * (filter_shape[i] - 1)
            + 1
            for i in range(dims)
        ]


@st.composite
def _roi_align_helper(draw):
    dtype = draw(helpers.get_dtypes("float", full=False))[0]
    N = draw(st.integers(1, 5))
    C = draw(st.integers(1, 5))
    H = W = draw(st.integers(5, 20))

    img_width = img_height = draw(st.integers(50, 100))

    spatial_scale = H / img_height

    output_size = draw(st.integers(H - 2, H + 5))

    sampling_ratio = draw(st.one_of(st.just(-1), st.integers(1, 3)))

    aligned = draw(st.booleans())
    input = draw(
        helpers.array_values(
            dtype=dtype,
            shape=(N, C, H, W),
            min_value=-3,
            max_value=3,
        )
    )
    bbox = {}
    for i in range(N):
        num_boxes = draw(st.integers(1, 5))
        for _ in range(num_boxes):
            x1 = draw(st.integers(0, img_width - 20))
            w = draw(st.integers(5, img_width - x1))
            y1 = draw(st.integers(0, img_height - 20))
            h = draw(st.integers(5, img_height - y1))
            bbox[(i, x1, y1, x1 + w, y1 + h)] = 1

    return (
        [dtype],
        input,
        np.array(list(bbox.keys()), dtype=dtype).reshape((-1, 5)),
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
    )


@st.composite
def _x_and_filters(
    draw,
    dim: int = 2,
    padding=None,
    transpose: bool = False,
    depthwise=False,
    general=False,
    bias=False,
):
    if not isinstance(dim, int):
        dim = draw(dim)
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(
        helpers.get_shape(
            min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
        )
    )
    dtype = draw(helpers.get_dtypes("float", full=False))
    input_channels = draw(st.integers(1, 3))
    output_channels = draw(st.integers(1, 3))
    group_list = [*range(1, 6)]
    if not transpose:
        group_list = list(
            filter(
                lambda x: (input_channels % x == 0 and x <= output_channels), group_list
            )
        )
    else:
        group_list = list(filter(lambda x: (output_channels % x == 0), group_list))
    fc = draw(st.sampled_from(group_list))
    strides = draw(
        st.one_of(
            st.integers(1, 3), st.lists(st.integers(1, 3), min_size=dim, max_size=dim)
        )
        if dim > 1
        else st.integers(1, 3)
    )
    dilations = draw(
        st.one_of(
            st.integers(1, 3), st.lists(st.integers(1, 3), min_size=dim, max_size=dim)
        )
        if dim > 1
        else st.integers(1, 3)
    )
    if dim == 2:
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
    elif dim == 1:
        data_format = draw(st.sampled_from(["NWC", "NCW"]))
    else:
        data_format = draw(st.sampled_from(["NDHWC", "NCDHW"]))
    fdilations = [dilations] * dim if isinstance(dilations, int) else dilations
    if padding is None:
        padding = st.one_of(
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=3),
                    st.integers(min_value=0, max_value=3),
                ),
                min_size=dim,
                max_size=dim,
            ),
            st.sampled_from(["SAME", "VALID"]),
            st.integers(min_value=0, max_value=3),
        )
    padding = draw(padding)
    if transpose:
        fstrides = [strides] * dim if isinstance(strides, int) else strides
        if isinstance(padding, list):
            assume(
                all(
                    max(pad) - min(pad) < min(stride, dilation)
                    for pad, stride, dilation in zip(padding, fstrides, fdilations)
                )
            )
        x_dim = draw(
            helpers.get_shape(
                min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
            )
        )
        output_shape = _output_shape(
            dim, fdilations, fstrides, padding, x_dim, filter_shape
        )
        assume(all(s > 0 for s in output_shape))
        if draw(st.booleans()):
            output_shape = None
    else:
        x_dim = []
        for i in range(dim):
            min_x = filter_shape[i] + (filter_shape[i] - 1) * (fdilations[i] - 1)
            x_dim.append(draw(st.integers(min_x, min_x + 1)))
        x_dim = tuple(x_dim)
    if not depthwise:
        if not transpose:
            output_channels = output_channels * fc
            filter_shape = filter_shape + (input_channels // fc, output_channels)
        else:
            input_channels = input_channels * fc
            filter_shape = filter_shape + (output_channels // fc, input_channels)
    else:
        filter_shape = filter_shape + (input_channels,)
    channel_first = True
    if data_format in ["NHWC", "NWC", "NDHWC"]:
        x_shape = (batch_size,) + x_dim + (input_channels,)
        channel_first = False
    else:
        x_shape = (batch_size, input_channels) + x_dim
    vals = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=x_shape,
            min_value=0.0,
            max_value=1.0,
        )
    )
    filters = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=filter_shape,
            min_value=0.0,
            max_value=1.0,
        )
    )
    if bias:
        bias_shape = (output_channels,)
        b = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=bias_shape,
                min_value=0.0,
                max_value=1.0,
            )
        )
    if general:
        data_format = "channel_first" if channel_first else "channel_last"
    if not transpose:
        x_dilation = draw(
            st.one_of(
                st.integers(1, 3),
                st.lists(st.integers(1, 3), min_size=dim, max_size=dim),
            )
        )
        dilations = (dilations, x_dilation)
    if not depthwise:
        filter_format = draw(st.sampled_from(["channel_first", "channel_last"]))
        if filter_format == "channel_first":
            filters = np.transpose(filters, (-1, -2, *range(dim)))
    ret = (
        dtype,
        vals,
        filters,
        dilations,
        data_format,
        strides,
        padding,
    )
    ret = ret + (output_shape, fc) if transpose else ret + (fc,)
    if not depthwise:
        ret = ret + (filter_format,)
    if bias:
        return ret + (b,)
    return ret


# output_shape not in conv_general_dilated
@st.composite
def _x_and_filters_and_transpose(
    draw,
    dim: int = 2,
    general=False,
    bias=False,
):
    transpose = draw(st.booleans())
    all_args = draw(
        _x_and_filters(
            dim=dim,
            general=general,
            bias=bias,
            transpose=transpose,
        )
    )
    output_shape = None
    if transpose:
        (
            dtype,
            x,
            filters,
            dilations,
            data_format,
            stride,
            pad,
            output_shape,
            fc,
            filter_format,
            bias,
        ) = all_args
    else:
        (
            dtype,
            x,
            filters,
            dilations,
            data_format,
            stride,
            pad,
            fc,
            filter_format,
            bias,
        ) = all_args
    return (
        dtype,
        x,
        filters,
        stride,
        pad,
        transpose,
        output_shape,
        data_format,
        filter_format,
        fc,
        dilations,
        bias,
    )


# Linear #
# -------#
@st.composite
def _x_and_linear(draw):
    mixed_fn_compos = draw(st.booleans())
    is_torch_backend = ivy.current_backend_str() == "torch"
    dtype = draw(
        # should sample from "valid" but with_supported_dtypes was not working
        helpers.get_dtypes("float", full=False, mixed_fn_compos=mixed_fn_compos)
    )
    in_features = draw(
        helpers.ints(min_value=1, max_value=2, mixed_fn_compos=mixed_fn_compos)
    )
    out_features = draw(
        helpers.ints(min_value=1, max_value=2, mixed_fn_compos=mixed_fn_compos)
    )

    x_shape = (
        1,
        1,
        in_features,
    )

    weight_shape = (1,) + (out_features,) + (in_features,)
    # if backend is torch and we're testing the primary implementation
    # weight.ndim should be equal to 2
    if is_torch_backend and not mixed_fn_compos:
        weight_shape = (out_features,) + (in_features,)

    bias_shape = (
        1,
        out_features,
    )

    x = draw(
        helpers.array_values(dtype=dtype[0], shape=x_shape, min_value=0, max_value=10)
    )
    weight = draw(
        helpers.array_values(
            dtype=dtype[0], shape=weight_shape, min_value=0, max_value=10
        )
    )
    bias = draw(
        helpers.array_values(
            dtype=dtype[0], shape=bias_shape, min_value=0, max_value=10
        )
    )
    return dtype, x, weight, bias


# LSTM #
# -----#


@st.composite
def _x_and_lstm(draw, dtypes):
    dtype = draw(dtypes)
    batch_shape = (1,)

    t = draw(helpers.ints(min_value=1, max_value=2))
    _in_ = draw(helpers.ints(min_value=1, max_value=2))
    _out_ = draw(helpers.ints(min_value=1, max_value=2))

    x_lstm_shape = batch_shape + (t,) + (_in_,)
    init_h_shape = batch_shape + (_out_,)
    init_c_shape = init_h_shape
    kernel_shape = (_in_,) + (4 * _out_,)
    recurrent_kernel_shape = (_out_,) + (4 * _out_,)
    bias_shape = (4 * _out_,)
    recurrent_bias_shape = bias_shape

    x_lstm = draw(
        helpers.array_values(
            dtype=dtype[0], shape=x_lstm_shape, min_value=0, max_value=1
        )
    )
    init_h = draw(
        helpers.array_values(
            dtype=dtype[0], shape=init_h_shape, min_value=0, max_value=1
        )
    )
    init_c = draw(
        helpers.array_values(
            dtype=dtype[0], shape=init_c_shape, min_value=0, max_value=1
        )
    )
    kernel = draw(
        helpers.array_values(
            dtype=dtype[0], shape=kernel_shape, min_value=0, max_value=1
        )
    )
    recurrent_kernel = draw(
        helpers.array_values(
            dtype=dtype[0], shape=recurrent_kernel_shape, min_value=0, max_value=1
        )
    )
    lstm_bias = draw(
        helpers.array_values(dtype=dtype[0], shape=bias_shape, min_value=0, max_value=1)
    )
    recurrent_bias = draw(
        helpers.array_values(
            dtype=dtype[0], shape=recurrent_bias_shape, min_value=0, max_value=1
        )
    )
    return (
        dtype,
        x_lstm,
        init_h,
        init_c,
        kernel,
        recurrent_kernel,
        lstm_bias,
        recurrent_bias,
    )


# Attention #
# ----------#


@st.composite
def _x_and_scaled_attention(draw, dtypes):
    dtype = draw(dtypes)
    num_queries = draw(helpers.ints(min_value=2, max_value=4))
    num_keys = draw(helpers.ints(min_value=2, max_value=4))
    feat_dim = draw(helpers.ints(min_value=2, max_value=4))
    batch_size = draw(helpers.ints(min_value=1, max_value=2))
    q_shape = (batch_size,) + (num_queries,) + (feat_dim,)
    k_shape = (batch_size,) + (num_keys,) + (feat_dim,)
    v_shape = (batch_size,) + (num_keys,) + (feat_dim,)
    mask_shape = (batch_size,) + (num_queries,) + (num_keys,)

    query = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=q_shape,
            min_value=1e-3,
            max_value=1e2,
            large_abs_safety_factor=7,
            small_abs_safety_factor=7,
            safety_factor_scale="linear",
        )
    )
    key = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=k_shape,
            min_value=1e-3,
            max_value=1e2,
            large_abs_safety_factor=7,
            small_abs_safety_factor=7,
            safety_factor_scale="linear",
        )
    )
    value = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=v_shape,
            min_value=1e-3,
            max_value=1e2,
            large_abs_safety_factor=7,
            small_abs_safety_factor=7,
            safety_factor_scale="linear",
        )
    )
    mask = draw(
        helpers.array_values(
            dtype="bool",
            shape=mask_shape,
        )
        | st.none()
    )
    return dtype, query, key, value, mask


# --- Main --- #
# ------------ #


# conv
@handle_test(
    fn_tree="functional.ivy.conv",
    dims=st.shared(st.integers(1, 3), key="dims"),
    x_f_d_df_tr=_x_and_filters_and_transpose(
        dim=st.shared(st.integers(1, 3), key="dims"),
        general=True,
        bias=True,
    ),
    ground_truth_backend="jax",
)
def test_conv(*, dims, x_f_d_df_tr, test_flags, backend_fw, fn_name, on_device):
    # pass
    (
        dtype,
        x,
        filters,
        stride,
        pad,
        transpose,
        output_shape,
        data_format,
        filter_format,
        fc,
        dilations,
        bias,
    ) = x_f_d_df_tr
    tf_dilations = dilations
    if not transpose:
        tf_dilations = tf_dilations[0]
        dilations, x_dilations = dilations
    else:
        x_dilations = None
    _assume_tf_dilation_gt_1(backend_fw, on_device, tf_dilations)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        transpose=transpose,
        dims=dims,
        output_shape=output_shape,
        data_format=data_format,
        filter_format=filter_format,
        feature_group_count=fc,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
    )


# conv1d
@handle_test(
    fn_tree="functional.ivy.conv1d",
    x_f_d_df=_x_and_filters(
        dim=1,
        bias=True,
    ),
    ground_truth_backend="jax",
)
def test_conv1d(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        fc,
        ff_format,
        bias,
    ) = x_f_d_df
    # ToDo: Enable gradient tests for dilations > 1 when tensorflow supports it.
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        filter_format=ff_format,
        x_dilations=dilations[1],
        dilations=dilations[0],
        bias=bias,
    )


# conv1d_transpose
@handle_test(
    fn_tree="functional.ivy.conv1d_transpose",
    x_f_d_df=_x_and_filters(
        dim=1,
        transpose=True,
        bias=True,
        padding=st.sampled_from(["SAME", "VALID"]),
    ),
    ground_truth_backend="torch",
)
def test_conv1d_transpose(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        output_shape,
        fc,
        filter_format,
        bias,
    ) = x_f_d_df
    # tensorflow does not work with dilations > 1 on cpu
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        filter_format=filter_format,
        data_format=data_format,
        dilations=dilations,
        bias=bias,
    )


# conv2d
@handle_test(
    fn_tree="functional.ivy.conv2d",
    x_f_d_df=_x_and_filters(
        dim=2,
        bias=True,
    ),
    ground_truth_backend="jax",
)
def test_conv2d(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        fc,
        ff_format,
        bias,
    ) = x_f_d_df
    # ToDo: Enable gradient tests for dilations > 1 when tensorflow supports it.
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        filter_format=ff_format,
        x_dilations=dilations[1],
        dilations=dilations[0],
        bias=bias,
    )


# conv2d_transpose
@handle_test(
    fn_tree="functional.ivy.conv2d_transpose",
    x_f_d_df=_x_and_filters(
        dim=2,
        transpose=True,
        bias=True,
    ),
    ground_truth_backend="torch",
)
def test_conv2d_transpose(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        output_shape,
        fc,
        filter_format,
        bias,
    ) = x_f_d_df
    assume(isinstance(pad, str) or backend_fw in ["torch", "tensorflow"])
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        filter_format=filter_format,
        data_format=data_format,
        dilations=dilations,
        bias=bias,
    )


# conv3d
@handle_test(
    fn_tree="functional.ivy.conv3d",
    x_f_d_df=_x_and_filters(
        dim=3,
        bias=True,
    ),
    ground_truth_backend="jax",
)
def test_conv3d(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        fc,
        ff_format,
        bias,
    ) = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        filter_format=ff_format,
        x_dilations=dilations[1],
        dilations=dilations[0],
        bias=bias,
    )


# conv3d_transpose
@handle_test(
    fn_tree="functional.ivy.conv3d_transpose",
    x_f_d_df=_x_and_filters(
        dim=3,
        transpose=True,
        bias=True,
        padding=st.sampled_from(["SAME", "VALID"]),
    ),
    ground_truth_backend="torch",
)
def test_conv3d_transpose(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        output_shape,
        fc,
        filter_format,
        bias,
    ) = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        filter_format=filter_format,
        data_format=data_format,
        dilations=dilations,
        bias=bias,
    )


# conv_general_dilated
@handle_test(
    fn_tree="functional.ivy.conv_general_dilated",
    dims=st.shared(st.integers(1, 3), key="dims"),
    x_f_d_df=_x_and_filters(
        dim=st.shared(st.integers(1, 3), key="dims"),
        general=True,
        bias=True,
    ),
    ground_truth_backend="torch",
)
def test_conv_general_dilated(
    *, dims, x_f_d_df, test_flags, backend_fw, fn_name, on_device
):
    (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        fc,
        ff_format,
        bias,
    ) = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        dims=dims,
        data_format=data_format,
        filter_format=ff_format,
        feature_group_count=fc,
        x_dilations=dilations[1],
        dilations=dilations[0],
        bias=bias,
    )


@handle_test(
    fn_tree="functional.ivy.conv_general_transpose",
    dim_x_f_d_df=_general_transpose_helper(),
    ground_truth_backend="torch",
)
def test_conv_general_transpose(
    *, dim_x_f_d_df, test_flags, backend_fw, fn_name, on_device
):
    dims, (
        dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        output_shape,
        fc,
        filter_format,
        bias,
    ) = dim_x_f_d_df
    assume(isinstance(pad, str) or backend_fw in ["torch", "tensorflow"])
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        dims=dims,
        filter_format=filter_format,
        data_format=data_format,
        output_shape=output_shape,
        dilations=dilations,
        feature_group_count=fc,
        bias=bias,
    )


# depthwise_conv2d
@handle_test(
    fn_tree="functional.ivy.depthwise_conv2d",
    x_f_d_df=_x_and_filters(
        dim=2,
        depthwise=True,
    ),
    # tensorflow does not support dilations > 1 and stride > 1
    ground_truth_backend="jax",
)
def test_depthwise_conv2d(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    dtype, x, filters, dilations, data_format, stride, pad, fc = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])
    # tensorflow only supports equal length strides in row and column
    if backend_fw == "tensorflow" and isinstance(stride, list) and len(stride) > 1:
        assume(stride[0] == stride[1])
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations[0],
    )


# dropout
@handle_test(
    fn_tree="functional.ivy.dropout",
    data=_dropout_helper(),
    test_gradients=st.just(False),
)
def test_dropout(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    (x_dtype, x), noise_shape, seed, dtype, prob, scale, training = data
    if not training or prob == 0:
        helpers.test_function(
            input_dtypes=x_dtype,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            on_device=on_device,
            x=x[0],
            prob=prob,
            scale=scale,
            noise_shape=noise_shape,
            dtype=dtype[0],
            training=training,
            seed=seed,
        )
    else:
        ret, gt_ret = helpers.test_function(
            input_dtypes=x_dtype,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            on_device=on_device,
            test_values=False,
            x=x[0],
            prob=prob,
            scale=scale,
            noise_shape=noise_shape,
            dtype=dtype[0],
            training=training,
            seed=seed,
        )
        ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
        gt_ret = helpers.flatten_and_to_np(
            ret=gt_ret, backend=test_flags.ground_truth_backend
        )
        for u, v, w in zip(ret, gt_ret, x):
            # cardinality test
            assert u.shape == v.shape == w.shape


# linear
@handle_test(
    fn_tree="functional.ivy.linear",
    dtype_x_weight_bias=_x_and_linear(),
)
def test_linear(*, dtype_x_weight_bias, test_flags, backend_fw, fn_name, on_device):
    dtype, x, weight, bias = dtype_x_weight_bias
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x,
        weight=weight,
        bias=bias,
    )


# TODO: fix this test
# lstm
# @handle_test(
#     fn_tree="functional.ivy.lstm",
#     dtypes_kwargs=_lstm_helper(),
#     ground_truth_backend="torch",
#     test_with_out=st.just(False),
# )
# def test_lstm(*, dtypes_kwargs, test_flags, backend_fw, fn_name, on_device):
#     dtypes, kwargs = dtypes_kwargs
#     assume("batch_sizes" not in kwargs)
#     helpers.test_function(
#         input_dtypes=dtypes,
#         test_flags=test_flags,
#         backend_to_test=backend_fw,
#         fn_name=fn_name,
#         on_device=on_device,
#         rtol_=1e-01,
#         atol_=1e-01,
#         **kwargs,
#     )


# lstm_update
@handle_test(
    fn_tree="functional.ivy.lstm_update",
    dtype_lstm=_x_and_lstm(
        dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_lstm_update(*, dtype_lstm, test_flags, backend_fw, fn_name, on_device):
    (
        dtype,
        x_lstm,
        init_h,
        init_c,
        kernel,
        recurrent_kernel,
        bias,
        recurrent_bias,
    ) = dtype_lstm
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-01,
        atol_=1e-01,
        x=x_lstm,
        init_h=init_h,
        init_c=init_c,
        kernel=kernel,
        recurrent_kernel=recurrent_kernel,
        bias=bias,
        recurrent_bias=recurrent_bias,
    )


# multi_head_attention
@handle_test(
    fn_tree="functional.ivy.multi_head_attention",
    dtype_mha=_mha_helper(),
    ground_truth_backend="numpy",
    # ToDo: fix the gradients and the container methods
    test_gradients=st.just(False),
    container_flags=st.just([False]),
)
def test_multi_head_attention(
    *,
    dtype_mha,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    (
        dtype,
        q,
        k,
        v,
        num_heads,
        attention_mask,
        in_proj_weights,
        q_proj_weights,
        k_proj_weights,
        v_proj_weights,
        out_proj_weights,
        in_proj_bias,
        out_proj_bias,
        key_padding_mask,
        bias_k,
        bias_v,
        static_k,
        static_v,
        scale,
        add_zero_attn,
        dropout,
        training,
        is_causal,
        return_attention_weights,
        average_attention_weights,
        batch_first,
    ) = dtype_mha
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        test_values=(dropout == 0),
        atol_=1e-02,
        rtol_=1e-02,
        query=q,
        key=k,
        value=v,
        batch_first=batch_first,
        num_heads=num_heads,
        scale=scale,
        attention_mask=attention_mask,
        in_proj_weights=in_proj_weights,
        q_proj_weights=q_proj_weights,
        k_proj_weights=k_proj_weights,
        v_proj_weights=v_proj_weights,
        out_proj_weights=out_proj_weights,
        in_proj_bias=in_proj_bias,
        out_proj_bias=out_proj_bias,
        is_causal=is_causal,
        key_padding_mask=key_padding_mask,
        bias_k=bias_k,
        bias_v=bias_v,
        static_k=static_k,
        static_v=static_v,
        add_zero_attn=add_zero_attn,
        return_attention_weights=return_attention_weights,
        average_attention_weights=average_attention_weights,
        dropout=dropout,
        training=training,
    )


@handle_test(
    fn_tree="functional.ivy.nms",
    inputs=_nms_helper(),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_nms(
    *,
    inputs,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    boxes, scores, iou_threshold, max_output_size, score_threshold = inputs
    helpers.test_function(
        input_dtypes=[ivy.float32, ivy.float32],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        boxes=boxes,
        scores=scores,
        iou_threshold=iou_threshold,
        max_output_size=max_output_size,
        score_threshold=score_threshold,
    )


@handle_test(
    fn_tree="functional.ivy.roi_align",
    inputs=_roi_align_helper(),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    ground_truth_backend="torch",
)
def test_roi_align(
    *,
    inputs,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtypes, input, boxes, output_size, spatial_scale, sampling_ratio, aligned = inputs

    helpers.test_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        input=input,
        boxes=boxes,
        output_size=output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
        rtol_=1e-5,
        atol_=1e-5,
    )


# scaled_dot_product_attention
@handle_test(
    fn_tree="functional.ivy.scaled_dot_product_attention",
    dtype_q_k_v_mask=_x_and_scaled_attention(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    scale=st.floats(min_value=0.1, max_value=1),
    dropout_p=st.floats(min_value=0, max_value=0.99),
    is_causal=st.booleans(),
    training=st.just(False),  # st.booleans(), disabled until proper testing is used
    ground_truth_backend="jax",
    test_with_out=st.just(True),
)
def test_scaled_dot_product_attention(
    *,
    dtype_q_k_v_mask,
    scale,
    dropout_p,
    is_causal,
    training,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    (dtype, query, key, value, mask) = dtype_q_k_v_mask
    is_causal = is_causal if mask is None else False
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-02,
        rtol_=1e-02,
        query=query,
        key=key,
        value=value,
        scale=scale,
        mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        training=training,
    )
