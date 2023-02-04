"""Collection of tests for unified neural network layers."""

# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
from ivy.functional.ivy.layers import _deconv_length

# Linear #
# -------#


@st.composite
def x_and_linear(draw, dtypes):
    dtype = draw(dtypes)
    in_features = draw(helpers.ints(min_value=1, max_value=2))
    out_features = draw(helpers.ints(min_value=1, max_value=2))

    x_shape = (
        1,
        1,
        in_features,
    )
    weight_shape = (1,) + (out_features,) + (in_features,)
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
    dtype_x_weight_bias=x_and_linear(
        dtypes=helpers.get_dtypes("numeric", full=False),
    ),
    ground_truth_backend="jax",
)
def test_linear(
    *,
    dtype_x_weight_bias,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, weight, bias = dtype_x_weight_bias
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
    shape = draw(helpers.get_shape(min_num_dims=1))
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
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
    return dtype_and_x, noise_shape


# dropout
@handle_test(
    fn_tree="functional.ivy.dropout",
    dtype_x_noiseshape=_dropout_helper(),
    prob=helpers.floats(min_value=0, max_value=0.9),
    scale=st.booleans(),
    training=st.booleans(),
    seed=helpers.ints(min_value=0, max_value=100),
    dtype=helpers.get_dtypes("float", full=False),
    test_gradients=st.just(False),
)
def test_dropout(
    *,
    dtype_x_noiseshape,
    prob,
    scale,
    training,
    seed,
    dtype,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    (x_dtype, x), noise_shape = dtype_x_noiseshape
    ret, gt_ret = helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=x_dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
    ret = helpers.flatten_and_to_np(ret=ret)
    gt_ret = helpers.flatten_and_to_np(ret=gt_ret)
    for u, v, w in zip(ret, gt_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape


# Attention #
# ----------#


@st.composite
def x_and_scaled_attention(draw, dtypes):
    dtype = draw(dtypes)
    num_queries = draw(helpers.ints(min_value=1, max_value=2))
    num_keys = draw(helpers.ints(min_value=1, max_value=2))
    feat_dim = draw(helpers.ints(min_value=1, max_value=2))
    scale = draw(helpers.floats(min_value=0.1, max_value=1))

    q_shape = (1,) + (num_queries,) + (feat_dim,)
    k_shape = (1,) + (num_keys,) + (feat_dim,)
    v_shape = (1,) + (num_keys,) + (feat_dim,)
    mask_shape = (1,) + (num_queries,) + (num_keys,)

    q = draw(
        helpers.array_values(dtype=dtype[0], shape=q_shape, min_value=0, max_value=10)
    )
    k = draw(
        helpers.array_values(dtype=dtype[0], shape=k_shape, min_value=0, max_value=10)
    )
    v = draw(
        helpers.array_values(dtype=dtype[0], shape=v_shape, min_value=0, max_value=10)
    )
    mask = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=mask_shape,
            min_value=0,
            max_value=10,
            large_abs_safety_factor=2,
            safety_factor_scale="linear",
        )
    )
    return dtype, q, k, v, mask, scale


# scaled_dot_product_attention
@handle_test(
    fn_tree="functional.ivy.scaled_dot_product_attention",
    dtype_q_k_v_mask_scale=x_and_scaled_attention(
        dtypes=helpers.get_dtypes("numeric", full=False),
    ),
    ground_truth_backend="jax",
)
def test_scaled_dot_product_attention(
    *,
    dtype_q_k_v_mask_scale,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, q, k, v, mask, scale = dtype_q_k_v_mask_scale
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        q=q,
        k=k,
        v=v,
        scale=scale,
        mask=mask,
    )


@st.composite
def x_and_mha(draw, dtypes):
    dtype = draw(dtypes)
    num_queries = draw(helpers.ints(min_value=1, max_value=3))
    feat_dim = draw(helpers.ints(min_value=1, max_value=3))
    num_heads = draw(helpers.ints(min_value=1, max_value=3))
    num_keys = draw(helpers.ints(min_value=1, max_value=3))

    x_mha_shape = (num_queries,) + (feat_dim * num_heads,)
    context_shape = (num_keys,) + (2 * feat_dim * num_heads,)
    mask_shape = (num_queries,) + (num_keys,)
    scale = draw(helpers.floats(min_value=0.1, max_value=1))
    x_mha = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=x_mha_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    context = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=context_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    mask = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=mask_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    return dtype, x_mha, scale, num_heads, context, mask


# multi_head_attention
@handle_test(
    fn_tree="functional.ivy.multi_head_attention",
    dtype_mha=x_and_mha(
        dtypes=helpers.get_dtypes("float"),
    ),
    ground_truth_backend="jax",
)
def test_multi_head_attention(
    *,
    dtype_mha,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x_mha, scale, num_heads, context, mask = dtype_mha
    to_q_fn = lambda x_, v: x_
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-02,
        rtol_=1e-02,
        x=x_mha,
        scale=scale,
        num_heads=num_heads,
        context=context,
        mask=mask,
        to_q_fn=to_q_fn,
        to_kv_fn=to_q_fn,
        to_out_fn=to_q_fn,
        to_q_v=None,
        to_kv_v=None,
        to_out_v=None,
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
):
    if not isinstance(dim, int):
        dim = draw(dim)
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
        )
    )
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(
        helpers.get_shape(
            min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
        )
    )
    dtype = draw(helpers.get_dtypes("float", full=False))
    input_channels = draw(st.integers(1, 3))
    output_channels = draw(st.integers(1, 3))
    group_list = [i for i in range(1, 6)]
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
        if not transpose:
            x_dilation = draw(
                st.one_of(
                    st.integers(1, 3),
                    st.lists(st.integers(1, 3), min_size=dim, max_size=dim),
                )
            )
            dilations = (dilations, x_dilation)
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
    if bias:
        return ret + (b,)
    return ret


def _assume_tf_dilation_gt_1(backend_fw, on_device, dilations):
    if not isinstance(backend_fw, str):
        backend_fw = backend_fw.current_backend_str()
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
    x_f_d_df=x_and_filters(dim=1),
    ground_truth_backend="jax",
)
def test_conv1d(
    *,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, fc = x_f_d_df
    # ToDo: Enable gradient tests for dilations > 1 when tensorflow supports it.
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv1d_transpose
@handle_test(
    fn_tree="functional.ivy.conv1d_transpose",
    x_f_d_df=x_and_filters(dim=1, transpose=True),
    ground_truth_backend="jax",
)
def test_conv1d_transpose(
    *,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
        dilations=dilations,
    )


# conv2d
@handle_test(
    fn_tree="functional.ivy.conv2d",
    x_f_d_df=x_and_filters(dim=2),
    ground_truth_backend="jax",
)
def test_conv2d(
    *,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, fc = x_f_d_df
    # ToDo: Enable gradient tests for dilations > 1 when tensorflow supports it.
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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


# conv2d_transpose
@handle_test(
    fn_tree="functional.ivy.conv2d_transpose",
    x_f_d_df=x_and_filters(
        dim=2,
        transpose=True,
    ),
    # tensorflow does not work with dilations > 1 on cpu
    ground_truth_backend="jax",
)
def test_conv2d_transpose(
    *,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
        dilations=dilations,
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
def test_depthwise_conv2d(
    *,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, fc = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    # tensorflow only supports equal length strides in row and column
    if (
        "tensorflow" in backend_fw.__name__
        and isinstance(stride, list)
        and len(stride) > 1
    ):
        assume(stride[0] == stride[1])
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
    x_f_d_df=x_and_filters(dim=3),
    ground_truth_backend="jax",
)
def test_conv3d(
    *,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, fc = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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


# conv3d_transpose
@handle_test(
    fn_tree="functional.ivy.conv3d_transpose",
    x_f_d_df=x_and_filters(
        dim=3,
        transpose=True,
    ),
    ground_truth_backend="jax",
)
def test_conv3d_transpose(
    *,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
        dilations=dilations,
    )


@handle_test(
    fn_tree="functional.ivy.conv_general_dilated",
    dims=st.shared(st.integers(1, 3), key="dims"),
    x_f_d_df=x_and_filters(
        dim=st.shared(st.integers(1, 3), key="dims"), general=True, bias=True
    ),
    ground_truth_backend="jax",
)
def test_conv_general_dilated(
    *,
    dims,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x, filters, dilations, data_format, stride, pad, fc, bias = x_f_d_df
    _assume_tf_dilation_gt_1(backend_fw, on_device, dilations[0])
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
    *,
    dims,
    x_f_d_df,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
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
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
        dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_lstm_update(
    *,
    dtype_lstm,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
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
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        fw=backend_fw,
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
