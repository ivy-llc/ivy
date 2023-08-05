"""Collection of tests for unified neural network layers."""

# global
from hypothesis import strategies as st, assume
import ivy
import numpy as np


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
from ivy.functional.ivy.layers import _deconv_length


# Linear #
# -------#
@st.composite
def x_and_linear(draw):
    mixed_fn_compos = draw(st.booleans())
    is_torch_backend = ivy.current_backend_str() == "torch"
    dtype = draw(
        helpers.get_dtypes("numeric", full=False, mixed_fn_compos=mixed_fn_compos)
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


# linear
@handle_test(
    fn_tree="functional.ivy.linear",
    dtype_x_weight_bias=x_and_linear(),
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


# Attention #
# ----------#


@st.composite
def x_and_scaled_attention(draw, dtypes):
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
            min_value=0,
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
            min_value=0,
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
            min_value=0,
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


# scaled_dot_product_attention
@handle_test(
    fn_tree="functional.ivy.scaled_dot_product_attention",
    dtype_q_k_v_mask=x_and_scaled_attention(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    scale=st.floats(min_value=0.1, max_value=1),
    dropout_p=st.floats(min_value=0, max_value=0.99),
    is_causal=st.booleans(),
    training=st.just(False),  # st.booleans(), disabled until proper testing is used
    ground_truth_backend="jax",
    test_with_out=st.just(True)
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
        atol_=1e-01,
        rtol_=1e-01,
        query=query,
        key=key,
        value=value,
        scale=scale,
        mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        training=training,
    )


@st.composite
def _mha_helper(draw):
    _qkv_same_dim = draw(st.booleans())
    _self_attention = draw(st.booleans())
    num_heads = draw(helpers.ints(min_value=1, max_value=3))
    _embed_dim = draw(helpers.ints(min_value=4, max_value=16)) * num_heads
    _num_queries = draw(helpers.ints(min_value=2, max_value=8))
    _num_keys = draw(helpers.ints(min_value=2, max_value=8))
    _batch_dim = draw(st.sampled_from([(), (1,)]))
    dtype = draw(helpers.get_dtypes("float", full=False, prune_function=False))
    in_proj_bias = None
    in_proj_weights = None
    q_proj_weights = None
    k_proj_weights = None
    v_proj_weights = None
    _mask_shape = (
        _num_queries,
        _num_queries if _self_attention and _qkv_same_dim else _num_keys,
    )
    if _qkv_same_dim:
        _pre_embed_dim = draw(helpers.ints(min_value=4, max_value=16))
        _q_shape = _batch_dim + (_num_queries, _pre_embed_dim)
        _kv_shape = _batch_dim + (_num_keys, _pre_embed_dim)

        q = draw(
            helpers.array_values(
                shape=_q_shape,
                dtype=dtype[0],
                large_abs_safety_factor=7,
                small_abs_safety_factor=7,
                safety_factor_scale="linear",
            )
        )
        k = draw(
            helpers.array_values(
                shape=_kv_shape,
                dtype=dtype[0],
                large_abs_safety_factor=7,
                small_abs_safety_factor=7,
                safety_factor_scale="linear",
            )
            if not _self_attention
            else st.none()
        )
        v = draw(
            helpers.array_values(
                shape=_kv_shape,
                dtype=dtype[0],
                large_abs_safety_factor=7,
                small_abs_safety_factor=7,
                safety_factor_scale="linear",
            )
            if not _self_attention
            else st.none()
        )
        in_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(3 * _embed_dim, _pre_embed_dim),
                min_value=0,
                max_value=10,
            )
            if _pre_embed_dim != _embed_dim
            else st.none()
        )
    else:
        _q_dim = draw(helpers.ints(min_value=2, max_value=8))
        _k_dim = draw(helpers.ints(min_value=2, max_value=8))
        _v_dim = draw(helpers.ints(min_value=2, max_value=8))
        _q_shape = _batch_dim + (_num_queries, _q_dim)
        _k_shape = _batch_dim + (_num_keys, _k_dim)
        _v_shape = _batch_dim + (_num_keys, _v_dim)
        q = draw(
            helpers.array_values(
                shape=_q_shape,
                dtype=dtype[0],
                large_abs_safety_factor=7,
                small_abs_safety_factor=7,
                safety_factor_scale="linear",
            )
        )
        k = draw(
            helpers.array_values(
                shape=_k_shape,
                dtype=dtype[0],
                large_abs_safety_factor=7,
                small_abs_safety_factor=7,
                safety_factor_scale="linear",
            )
        )
        v = draw(
            helpers.array_values(
                shape=_v_shape,
                dtype=dtype[0],
                large_abs_safety_factor=7,
                small_abs_safety_factor=7,
                safety_factor_scale="linear",
            )
        )
        q_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_embed_dim, _q_dim),
                min_value=0,
                max_value=2,
            )
        )
        k_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_embed_dim, _k_dim),
                min_value=0,
                max_value=2,
            )
        )
        v_proj_weights = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=(_embed_dim, _v_dim),
                min_value=0,
                max_value=2,
            )
        )

    in_proj_bias = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(3 * _embed_dim), min_value=0, max_value=10
        )
        | st.none()
    )
    _out_dim = draw(helpers.ints(min_value=4, max_value=16))
    out_proj_weights = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(_out_dim, _embed_dim),
            min_value=0,
            max_value=2,
        )
        | st.none()
    )
    out_proj_bias = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(_out_dim), min_value=0, max_value=10
        )
        | st.none()
    )
    attention_mask = draw(
        helpers.array_values(
            dtype="bool",
            shape=_mask_shape,
        )
        | st.none()
    )
    return (
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
    )


# multi_head_attention
@handle_test(
    fn_tree="functional.ivy.multi_head_attention",
    dtype_mha=_mha_helper(),
    scale=st.one_of(st.floats(), st.none()),
    dropout=st.floats(min_value=0, max_value=0.99),
    training=st.just(False),  # st.booleans(), disabled until proper testing is used
    is_causal=st.booleans(),
    return_attention_weights=st.booleans(),
    average_attention_weights=st.booleans(),
    ground_truth_backend="jax",
)
def test_multi_head_attention(
    *,
    dtype_mha,
    scale,
    dropout,
    training,
    is_causal,
    return_attention_weights,
    average_attention_weights,
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
    ) = dtype_mha
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-02,
        rtol_=1e-02,
        query=q,
        key=k,
        value=v,
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
        return_attention_weights=return_attention_weights,
        average_attention_weights=average_attention_weights,
        dropout=dropout,
        training=training,
    )


# Convolutions #
# -------------#


@st.composite
def x_and_filters(
    draw,
    dim: int = 2,
    transpose: bool = False,
    depthwise=False,
    general=False,
    bias=False,
    filter_format=None,
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
        group_list = list(filter(lambda x: (input_channels % x == 0), group_list))
    else:
        group_list = list(filter(lambda x: (output_channels % x == 0), group_list))
    fc = draw(st.sampled_from(group_list)) if general else 1
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

    full_strides = [strides] * dim if isinstance(strides, int) else strides
    full_dilations = [dilations] * dim if isinstance(dilations, int) else dilations
    if transpose:
        padding = draw(st.sampled_from(["SAME", "VALID"]))
        x_dim = draw(
            helpers.get_shape(
                min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
            )
        )
        if draw(st.booleans()):
            output_shape = []
            for i in range(dim):
                output_shape.append(
                    _deconv_length(
                        x_dim[i],
                        full_strides[i],
                        filter_shape[i],
                        padding,
                        full_dilations[i],
                    )
                )
        else:
            output_shape = None
    else:
        padding = draw(
            st.one_of(
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
        )
        x_dim = []
        for i in range(dim):
            min_x = filter_shape[i] + (filter_shape[i] - 1) * (full_dilations[i] - 1)
            x_dim.append(draw(st.integers(min_x, min_x + 1)))
        x_dim = tuple(x_dim)
    if not depthwise:
        if not transpose:
            output_channels = output_channels * fc
            filter_shape = filter_shape + (input_channels // fc, output_channels)
        else:
            input_channels = input_channels * fc
            filter_shape = filter_shape + (input_channels, output_channels // fc)
    else:
        filter_shape = filter_shape + (input_channels,)
    channel_first = True
    if data_format == "NHWC" or data_format == "NWC" or data_format == "NDHWC":
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

    x_dilation = draw(
        st.one_of(
            st.integers(1, 3),
            st.lists(st.integers(1, 3), min_size=dim, max_size=dim),
        )
    )
    dilations = (dilations, x_dilation)
    if filter_format is not None:
        filter_format = draw(filter_format)
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
    ret = ret + (filter_format,) if filter_format is not None else ret
    if bias:
        return ret + (b,)
    return ret


def _assume_tf_dilation_gt_1(backend_fw, on_device, dilations):
    if backend_fw == "tensorflow":
        assume(
            not (
                on_device == "cpu" and (dilations > 1)
                if isinstance(dilations, int)
                else any(d > 1 for d in dilations)
            )
        )


# conv1d
@handle_test(
    fn_tree="functional.ivy.conv1d",
    x_f_d_df=x_and_filters(
        dim=1,
        bias=True,
        filter_format=st.sampled_from(["channel_last", "channel_first"]),
    ),
    ground_truth_backend="jax",
)
def test_conv1d(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    dtype, x, filters, dilations, data_format, stride, pad, fc, ff_format, bias = (
        x_f_d_df
    )
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
    x_f_d_df=x_and_filters(
        dim=1,
        transpose=True,
        bias=True,
    ),
    ground_truth_backend="jax",
)
def test_conv1d_transpose(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc, bias = (
        x_f_d_df
    )
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        # tensorflow does not work with dilations > 1 on cpu
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations[0],
        bias=bias,
    )


# conv2d
@handle_test(
    fn_tree="functional.ivy.conv2d",
    x_f_d_df=x_and_filters(
        dim=2,
        bias=True,
        filter_format=st.sampled_from(["channel_last", "channel_first"]),
    ),
    ground_truth_backend="jax",
)
def test_conv2d(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    dtype, x, filters, dilations, data_format, stride, pad, fc, ff_format, bias = (
        x_f_d_df
    )
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
    x_f_d_df=x_and_filters(
        dim=2,
        transpose=True,
        bias=True,
    ),
    # tensorflow does not work with dilations > 1 on cpu
    ground_truth_backend="jax",
)
def test_conv2d_transpose(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc, bias = (
        x_f_d_df
    )
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])

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
        data_format=data_format,
        dilations=dilations[0],
        bias=bias,
    )


# depthwise_conv2d
@handle_test(
    fn_tree="functional.ivy.depthwise_conv2d",
    x_f_d_df=x_and_filters(
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
        dilations=dilations,
    )


# conv3d
@handle_test(
    fn_tree="functional.ivy.conv3d",
    x_f_d_df=x_and_filters(
        dim=3,
        bias=True,
        filter_format=st.sampled_from(["channel_last", "channel_first"]),
    ),
    ground_truth_backend="jax",
)
def test_conv3d(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    dtype, x, filters, dilations, data_format, stride, pad, fc, ff_format, bias = (
        x_f_d_df
    )
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
    x_f_d_df=x_and_filters(
        dim=3,
        transpose=True,
        bias=True,
    ),
    ground_truth_backend="jax",
)
def test_conv3d_transpose(*, x_f_d_df, test_flags, backend_fw, fn_name, on_device):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc, bias = (
        x_f_d_df
    )
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
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations[0],
        bias=bias,
    )


# conv_general_dilated
@handle_test(
    fn_tree="functional.ivy.conv_general_dilated",
    dims=st.shared(st.integers(1, 3), key="dims"),
    x_f_d_df=x_and_filters(
        dim=st.shared(st.integers(1, 3), key="dims"),
        general=True,
        bias=True,
        filter_format=st.sampled_from(["channel_last", "channel_first"]),
    ),
    ground_truth_backend="jax",
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
    dims=st.shared(st.integers(1, 3), key="dims"),
    x_f_d_df=x_and_filters(
        dim=st.shared(st.integers(1, 3), key="dims"),
        general=True,
        transpose=True,
        bias=True,
    ),
    ground_truth_backend="jax",
)
def test_conv_general_transpose(
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
        output_shape,
        fc,
        bias,
    ) = x_f_d_df
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
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        feature_group_count=fc,
        bias=bias,
    )


# LSTM #
# -----#


@st.composite
def x_and_lstm(draw, dtypes):
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


# lstm
@handle_test(
    fn_tree="functional.ivy.lstm_update",
    dtype_lstm=x_and_lstm(
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
