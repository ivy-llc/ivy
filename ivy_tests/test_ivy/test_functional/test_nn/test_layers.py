"""Collection of tests for unified neural network layers."""

# global
import numpy as np
from hypothesis import given, strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# Linear #
# -------#


@st.composite
def x_and_linear(draw, dtypes):
    dtype = draw(dtypes)
    outer_batch_shape = draw(
        st.tuples(
            helpers.ints(min_value=3, max_value=5),
            helpers.ints(min_value=1, max_value=3),
            helpers.ints(min_value=1, max_value=3),
        )
    )
    inner_batch_shape = draw(
        st.tuples(
            helpers.ints(min_value=3, max_value=5),
            helpers.ints(min_value=1, max_value=3),
            helpers.ints(min_value=1, max_value=3),
        )
    )
    in_features = draw(helpers.ints(min_value=1, max_value=3))
    out_features = draw(helpers.ints(min_value=1, max_value=3))

    x_shape = outer_batch_shape + inner_batch_shape + (in_features,)
    weight_shape = outer_batch_shape + (out_features,) + (in_features,)
    bias_shape = outer_batch_shape + (out_features,)

    x = draw(helpers.array_values(dtype=dtype, shape=x_shape, min_value=0, max_value=1))
    weight = draw(
        helpers.array_values(dtype=dtype, shape=weight_shape, min_value=0, max_value=1)
    )
    bias = draw(
        helpers.array_values(dtype=dtype, shape=bias_shape, min_value=0, max_value=1)
    )
    return dtype, x, weight, bias


# linear
@given(
    dtype_x_weight_bias=x_and_linear(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
    ),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="linear"),
    data=st.data(),
)
@handle_cmd_line_args
def test_linear(
    *,
    data,
    dtype_x_weight_bias,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):

    dtype, x, weight, bias = dtype_x_weight_bias
    as_variable = [as_variable, as_variable, as_variable]
    native_array = [native_array, native_array, native_array]
    container = [container, container, container]

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="linear",
        rtol_=1e-03,
        atol_=1e-03,
        x=np.asarray(x, dtype=dtype),
        weight=np.asarray(weight, dtype=dtype),
        bias=np.asarray(bias, dtype=dtype),
    )


# Dropout #
# --------#

# dropout
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    data=st.data(),
    prob=st.floats(min_value=0, max_value=0.9, width=64),
    scale=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="dropout"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
@handle_cmd_line_args
def test_dropout(
    *,
    data,
    dtype_and_x,
    prob,
    scale,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):

    dtype, x = dtype_and_x
    x = np.asarray(x, dtype=dtype)
    ret = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="dropout",
        test_values=False,
        x=x,
        prob=prob,
        scale=scale,
        dtype=dtype,
    )
    ret = helpers.flatten(ret=ret)
    for u in ret:
        # cardinality test
        assert u.shape == x.shape


# Attention #
# ----------#


@st.composite
def x_and_scaled_attention(draw, dtypes):
    dtype = draw(dtypes)
    batch_shape = draw(
        st.tuples(
            helpers.ints(min_value=3, max_value=5),
            helpers.ints(min_value=1, max_value=3),
            helpers.ints(min_value=1, max_value=3),
        )
    )
    num_queries = draw(helpers.ints(min_value=1, max_value=3))
    num_keys = draw(helpers.ints(min_value=1, max_value=3))
    feat_dim = draw(helpers.ints(min_value=1, max_value=3))
    scale = draw(st.floats(min_value=0.1, max_value=1, width=64))

    q_shape = batch_shape + (num_queries,) + (feat_dim,)
    k_shape = batch_shape + (num_keys,) + (feat_dim,)
    v_shape = batch_shape + (num_keys,) + (feat_dim,)
    mask_shape = batch_shape + (num_queries,) + (num_keys,)

    q = draw(helpers.array_values(dtype=dtype, shape=q_shape, min_value=0, max_value=1))
    k = draw(helpers.array_values(dtype=dtype, shape=k_shape, min_value=0, max_value=1))
    v = draw(helpers.array_values(dtype=dtype, shape=v_shape, min_value=0, max_value=1))
    mask = draw(
        helpers.array_values(
            dtype=dtype,
            shape=mask_shape,
            min_value=0,
            max_value=1,
            large_value_safety_factor=2,
        )
    )
    return dtype, q, k, v, mask, scale


# # scaled_dot_product_attention
@given(
    dtype_q_k_v_mask_scale=x_and_scaled_attention(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="scaled_dot_product_attention"
    ),
    data=st.data(),
)
@handle_cmd_line_args
def test_scaled_dot_product_attention(
    *,
    data,
    dtype_q_k_v_mask_scale,
    as_variable,
    num_positional_args,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, q, k, v, mask, scale = dtype_q_k_v_mask_scale
    dtype = [dtype] * 4
    as_variable = [as_variable] * 4
    native_array = [native_array] * 4
    container = [container] * 4

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="scaled_dot_product_attention",
        ground_truth_backend="jax",
        q=np.asarray(q, dtype=dtype[0]),
        k=np.asarray(k, dtype=dtype[0]),
        v=np.asarray(v, dtype=dtype[0]),
        scale=scale,
        mask=np.asarray(mask, dtype=dtype[0]),
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
    scale = draw(st.floats(min_value=0.1, max_value=1, width=64))
    x_mha = draw(
        helpers.array_values(
            dtype=dtype,
            shape=x_mha_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    context = draw(
        helpers.array_values(
            dtype=dtype,
            shape=context_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    mask = draw(
        helpers.array_values(
            dtype=dtype,
            shape=mask_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    return dtype, x_mha, scale, num_heads, context, mask


# multi_head_attention
@given(
    dtype_mha=x_and_mha(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="multi_head_attention"),
    data=st.data(),
)
@handle_cmd_line_args
def test_multi_head_attention(
    *,
    data,
    dtype_mha,
    as_variable,
    num_positional_args,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x_mha, scale, num_heads, context, mask = dtype_mha
    as_variable = [as_variable] * 3
    native_array = [native_array] * 3
    container = [container] * 3
    to_q_fn = lambda x_, v: x_

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="multi_head_attention",
        ground_truth_backend="jax",
        x=np.asarray(x_mha, dtype=dtype),
        scale=scale,
        num_heads=num_heads,
        context=np.asarray(context, dtype=dtype),
        mask=np.asarray(mask, dtype=dtype),
        to_q_fn=to_q_fn,
        to_kv_fn=to_q_fn,
        to_out_fn=to_q_fn,
        to_q_v=None,
        to_kv_v=None,
        to_out_v=None,
    )


# Convolutions #
# -------------#


def _deconv_length(dim_size, stride_size, kernel_size, padding, dilation=1):
    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    if padding == "VALID":
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == "SAME":
        dim_size = dim_size * stride_size

    return dim_size


@st.composite
def _x_and_filters(
    draw,
    dtypes,
    data_format,
    padding,
    stride_min,
    stride_max,
    type: str = "2d",
    transpose=False,
):
    data_format = draw(data_format)
    dtype = draw(dtypes)
    padding = draw(padding)
    stride = draw(helpers.ints(min_value=stride_min, max_value=stride_max))
    dilations = draw(helpers.ints(min_value=1, max_value=3))
    if type == "1d":
        if not transpose:
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=1, max_value=3),
                    helpers.ints(min_value=1, max_value=3),
                )
            )
            min_x_width = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
        else:
            filter_shape = draw(
                st.tuples(
                    st.ints(min_value=3, max_value=5),
                    st.shared(helpers.ints(min_value=1, max_value=3), key="d_in"),
                    st.shared(helpers.ints(min_value=1, max_value=3), key="d_in"),
                )
            )
            min_x_width = 1
        d_in = filter_shape[1]
        if data_format == "NWC":
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=min_x_width, max_value=100),
                    helpers.ints(min_value=d_in, max_value=d_in),
                )
            )
            x_w = x_shape[1]
        else:
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=d_in, max_value=d_in),
                    helpers.ints(min_value=min_x_width, max_value=100),
                )
            )
            x_w = x_shape[2]
        if transpose:
            output_shape = [
                _deconv_length(x_w, stride, filter_shape[0], padding, dilations)
            ]
    elif type == "2d" or type == "depthwise":
        min_x_height = 1
        min_x_width = 1
        if type == "depthwise":
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=1, max_value=3),
                    helpers.ints(min_value=1, max_value=3),
                )
            )
        elif not transpose:
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=1, max_value=3),
                    helpers.ints(min_value=1, max_value=3),
                )
            )
        else:
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=3, max_value=5),
                    st.shared(helpers.ints(min_value=1, max_value=3), key="d_in"),
                    st.shared(helpers.ints(min_value=1, max_value=3), key="d_in"),
                )
            )
        if not transpose:
            min_x_height = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
            min_x_width = filter_shape[1] + (filter_shape[1] - 1) * (dilations - 1)
        d_in = filter_shape[2]
        if data_format == "NHWC":
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=min_x_height, max_value=100),
                    helpers.ints(min_value=min_x_width, max_value=100),
                    helpers.ints(min_value=d_in, max_value=d_in),
                )
            )
            x_h = x_shape[1]
            x_w = x_shape[2]
        else:
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=d_in, max_value=d_in),
                    helpers.ints(min_value=min_x_height, max_value=100),
                    helpers.ints(min_value=min_x_width, max_value=100),
                )
            )
            x_h = x_shape[2]
            x_w = x_shape[3]
        if transpose:
            output_shape_h = _deconv_length(
                x_h, stride, filter_shape[0], padding, dilations
            )
            output_shape_w = _deconv_length(
                x_w, stride, filter_shape[1], padding, dilations
            )
            output_shape = [output_shape_h, output_shape_w]
    else:
        if not transpose:
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=1, max_value=3),
                    helpers.ints(min_value=1, max_value=3),
                )
            )
            min_x_depth = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
            min_x_height = filter_shape[1] + (filter_shape[1] - 1) * (dilations - 1)
            min_x_width = filter_shape[2] + (filter_shape[2] - 1) * (dilations - 1)
        else:
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=3, max_value=5),
                    st.shared(helpers.ints(min_value=1, max_value=3), key="d_in"),
                    st.shared(helpers.ints(min_value=1, max_value=3), key="d_in"),
                )
            )
            min_x_depth = 1
            min_x_height = 1
            min_x_width = 1
        d_in = filter_shape[3]
        if data_format == "NDHWC":
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=min_x_depth, max_value=100),
                    helpers.ints(min_value=min_x_height, max_value=100),
                    helpers.ints(min_value=min_x_width, max_value=100),
                    helpers.ints(min_value=d_in, max_value=d_in),
                )
            )
            x_d = x_shape[1]
            x_h = x_shape[2]
            x_w = x_shape[3]
        else:
            x_shape = draw(
                st.tuples(
                    helpers.ints(min_value=1, max_value=5),
                    helpers.ints(min_value=d_in, max_value=d_in),
                    helpers.ints(min_value=min_x_depth, max_value=100),
                    helpers.ints(min_value=min_x_width, max_value=100),
                    helpers.ints(min_value=min_x_width, max_value=100),
                )
            )
            x_d = x_shape[2]
            x_h = x_shape[3]
            x_w = x_shape[4]
        if transpose:
            output_shape_d = _deconv_length(
                x_d, stride, filter_shape[0], padding, dilations
            )
            output_shape_h = _deconv_length(
                x_h, stride, filter_shape[1], padding, dilations
            )
            output_shape_w = _deconv_length(
                x_w, stride, filter_shape[2], padding, dilations
            )
            output_shape = [output_shape_d, output_shape_h, output_shape_w]
    x = draw(helpers.array_values(dtype=dtype, shape=x_shape, min_value=0, max_value=1))
    filters = draw(
        helpers.array_values(dtype=dtype, shape=filter_shape, min_value=0, max_value=1)
    )
    if not transpose:
        return dtype, x, filters, dilations, data_format, stride, padding
    return dtype, x, filters, dilations, data_format, stride, padding, output_shape


# conv1d
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NWC", "NCW"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=4,
        type="1d",
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv1d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv1d(
    *,
    data,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    dtype = [dtype] * 2
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv1d",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv1d_transpose
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NWC", "NCW"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=4,
        type="1d",
        transpose=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv1d_transpose"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv1d_transpose(
    *,
    data,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    dtype = [dtype] * 2
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv1d_transpose",
        ground_truth_backend="jax",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# conv2d
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NHWC", "NCHW"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=4,
        type="2d",
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv2d(
    *,
    data,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    dtype = [dtype] * 2

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv2d",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv2d_transpose
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NHWC", "NCHW"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=4,
        type="2d",
        transpose=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d_transpose"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv2d_transpose(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    dtype = [dtype] * 2
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv2d_transpose",
        device_=device,
        ground_truth_backend="jax",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# depthwise_conv2d
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NHWC", "NCHW"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=4,
        type="depthwise",
    ),
    num_positional_args=helpers.num_positional_args(fn_name="depthwise_conv2d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_depthwise_conv2d(
    *,
    x_f_d_df,
    with_out,
    num_positional_args,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    assume(not (fw == "tensorflow" and dilations > 1 and stride > 1))
    dtype = [dtype] * 2
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="depthwise_conv2d",
        ground_truth_backend="jax",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv3d
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NDHWC", "NCDHW"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=4,
        type="3d",
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv3d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv3d(
    *,
    data,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    dtype = [dtype] * 2

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv3d",
        ground_truth_backend="jax",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv3d_transpose
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NDHWC", "NCDHW"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=4,
        type="3d",
        transpose=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv3d_transpose"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv3d_transpose(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    dtype = [dtype] * 2

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv3d_transpose",
        ground_truth_backend="jax",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# LSTM #
# -----#


@st.composite
def x_and_lstm(draw, dtypes):
    dtype = draw(dtypes)
    batch_shape = draw(
        st.tuples(
            helpers.ints(min_value=3, max_value=5),
            helpers.ints(min_value=1, max_value=3),
            helpers.ints(min_value=1, max_value=3),
        )
    )

    t = draw(helpers.ints(min_value=1, max_value=3))
    _in_ = draw(helpers.ints(min_value=1, max_value=3))
    _out_ = draw(helpers.ints(min_value=1, max_value=3))

    x_lstm_shape = batch_shape + (t,) + (_in_,)
    init_h_shape = batch_shape + (_out_,)
    init_c_shape = init_h_shape
    kernel_shape = (_in_,) + (4 * _out_,)
    recurrent_kernel_shape = (_out_,) + (4 * _out_,)
    bias_shape = (4 * _out_,)
    recurrent_bias_shape = bias_shape

    x_lstm = draw(
        helpers.array_values(dtype=dtype, shape=x_lstm_shape, min_value=0, max_value=1)
    )
    init_h = draw(
        helpers.array_values(dtype=dtype, shape=init_h_shape, min_value=0, max_value=1)
    )
    init_c = draw(
        helpers.array_values(dtype=dtype, shape=init_c_shape, min_value=0, max_value=1)
    )
    kernel = draw(
        helpers.array_values(dtype=dtype, shape=kernel_shape, min_value=0, max_value=1)
    )
    recurrent_kernel = draw(
        helpers.array_values(
            dtype=dtype, shape=recurrent_kernel_shape, min_value=0, max_value=1
        )
    )
    lstm_bias = draw(
        helpers.array_values(dtype=dtype, shape=bias_shape, min_value=0, max_value=1)
    )
    recurrent_bias = draw(
        helpers.array_values(
            dtype=dtype, shape=recurrent_bias_shape, min_value=0, max_value=1
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
@given(
    dtype_lstm=x_and_lstm(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="lstm_update"),
    data=st.data(),
)
@handle_cmd_line_args
def test_lstm(
    *,
    data,
    dtype_lstm,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
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
    as_variable = [as_variable] * 7
    native_array = [native_array] * 7
    container = [container] * 7

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="lstm_update",
        x=np.asarray(x_lstm, dtype=dtype),
        init_h=np.asarray(init_h, dtype=dtype),
        init_c=np.asarray(init_c, dtype=dtype),
        kernel=np.asarray(kernel, dtype=dtype),
        recurrent_kernel=np.asarray(recurrent_kernel, dtype=dtype),
        bias=np.asarray(bias, dtype=dtype),
        recurrent_bias=np.asarray(recurrent_bias, dtype=dtype),
    )
