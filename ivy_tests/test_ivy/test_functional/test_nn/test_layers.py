"""Collection of tests for unified neural network layers."""

# global
import pytest
import numpy as np
from hypothesis import given, assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# Linear #
# -------#


@st.composite
def x_and_weight(draw, dtypes, fn_name):
    dtype = draw(dtypes)
    outer_batch_shape = draw(
        st.tuples(
            st.integers(3, 5),
            st.integers(1, 3),
            st.integers(1, 3),
        )
    )
    inner_batch_shape = draw(
        st.tuples(
            st.integers(3, 5),
            st.integers(1, 3),
            st.integers(1, 3),
        )
    )
    batch_shape = inner_batch_shape

    #Linear
    in_features = draw(st.integers(min_value=1, max_value=3))
    out_features = draw(st.integers(min_value=1, max_value=3))

    x_shape = outer_batch_shape + inner_batch_shape + (in_features,)
    weight_shape = outer_batch_shape + (out_features,) + (in_features,)
    bias_shape = outer_batch_shape + (out_features,)

    x = draw(helpers.array_values
             (dtype=dtype, shape=x_shape, min_value=0, max_value=1))
    weight = draw(helpers.array_values
                  (dtype=dtype, shape=weight_shape, min_value=0, max_value=1))
    bias = draw(helpers.array_values
                (dtype=dtype, shape=bias_shape, min_value=0, max_value=1))

    #Scaled_dot_product_attention
    num_queries = in_features
    num_keys = in_features
    feat_dim = in_features
    scale = draw(
        st.floats(
            min_value=0.10000000149011612,
            max_value=1,
            width=64
        )
    )

    q_shape = batch_shape + (num_queries,) + (feat_dim,)
    k_shape = batch_shape + (num_keys,) + (feat_dim,)
    v_shape = batch_shape + (num_keys,) + (feat_dim,)
    mask_shape = batch_shape + (num_queries,) + (num_keys,)

    q = draw(helpers.array_values
             (dtype=dtype, shape=q_shape, min_value=0, max_value=1))
    k = draw(helpers.array_values
             (dtype=dtype, shape=k_shape, min_value=0, max_value=1))
    v = draw(helpers.array_values
             (dtype=dtype, shape=v_shape, min_value=0, max_value=1))
    mask = draw(helpers.array_values
                (dtype=dtype, shape=mask_shape, min_value=0, max_value=1, safety_factor=2))

    #Update_lstm
    t = draw(st.integers(min_value=1, max_value=3))
    _in_ = draw(st.integers(min_value=1, max_value=3))
    _out_ = draw(st.integers(min_value=1, max_value=3))

    x_lstm_shape = batch_shape + (t,) + (_in_,)
    init_h_shape = batch_shape + (_out_,)
    init_c_shape = init_h_shape
    kernel_shape = (_in_,) + (4 * _out_,)
    recurrent_kernel_shape = (_out_,) + (4 * _out_,)
    bias_shape = (4 * _out_,)
    recurrent_bias_shape = bias_shape

    x_lstm = draw(helpers.array_values
                  (dtype=dtype, shape=x_lstm_shape, min_value=0, max_value=1))
    init_h = draw(helpers.array_values
                  (dtype=dtype, shape=init_h_shape, min_value=0, max_value=1))
    init_c = draw(helpers.array_values
                  (dtype=dtype, shape=init_c_shape, min_value=0, max_value=1))
    kernel = draw(helpers.array_values
                  (dtype=dtype, shape=kernel_shape, min_value=0, max_value=1))
    recurrent_kernel = draw(helpers.array_values
                  (dtype=dtype, shape=recurrent_kernel_shape, min_value=0, max_value=1))
    lstm_bias = draw(helpers.array_values
                  (dtype=dtype, shape=bias_shape, min_value=0, max_value=1))
    recurrent_bias = draw(helpers.array_values
                  (dtype=dtype, shape=recurrent_bias_shape, min_value=0, max_value=1))

    #Multi_head_attention
    num_heads = num_keys

    x_mha = q
    context = k

    if fn_name == "linear":
        return dtype, x, weight, bias
    if fn_name == "scaled_dot_product_attention":
        return dtype, q, k, v, mask, scale
    if fn_name == "lstm_update":
        return dtype, x_lstm,init_h, init_c, kernel, recurrent_kernel, lstm_bias, recurrent_bias
    if fn_name == "multi_head_attention":
        return dtype, x_mha, scale, num_heads, context, mask


# linear
@given(
    dtype_x_weight_bias=x_and_weight(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        fn_name="linear",
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
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
    prob=st.floats(
        min_value=0.10000000149011612,
        max_value=1,
        width=64
    ),
    scale=st.booleans(),
    with_out=st.booleans(),
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
    dtype,
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

# # scaled_dot_product_attention
@given(
    dtype_q_k_v_mask_scale=x_and_weight(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        fn_name="scaled_dot_product_attention",
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
    as_variable = [as_variable] * 4
    native_array = [native_array] * 4
    container = [container] * 4

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="scaled_dot_product_attention",
        q=np.asarray(q, dtype=dtype),
        k=np.asarray(k, dtype=dtype),
        v=np.asarray(v, dtype=dtype),
        scale=scale,
        mask=np.asarray(mask, dtype=dtype),
    )


# multi_head_attention
@given(
    dtype_mha=x_and_weight(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        fn_name="multi_head_attention",
    ),
    to_q_fn = st.functions(like=lambda x,v:x),
    to_kv_fn = st.functions(like=lambda x,v:x),
    to_out_fn = st.functions(like=lambda x,v:x),
    with_out=st.booleans(),
    data=st.data(),
)
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
        x=np.asarray(x_mha, dtype=dtype),
        context=np.asarray(context, dtype=dtype),
        scale=scale,
        mask=np.asarray(mask, dtype=dtype),
    )



# Convolutions #
# -------------#


@st.composite
def x_and_filters(draw, dtypes, data_format, type: str = "2d"):
    data_format = draw(data_format)
    dtype = draw(dtypes)
    dilations = draw(st.integers(min_value=1, max_value=3))
    if type == "1d":
        filter_shape = draw(
            st.tuples(
                st.integers(3, 5),
                st.integers(1, 3),
                st.integers(1, 3),
            )
        )

        min_x_width = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
        d_in = filter_shape[1]
        if data_format == "NWC":
            x_shape = draw(
                st.tuples(
                    st.integers(1, 5),
                    st.integers(min_value=min_x_width, max_value=100),
                    st.integers(d_in, d_in),
                )
            )
        else:
            x_shape = draw(
                st.tuples(
                    st.integers(1, 5),
                    st.integers(d_in, d_in),
                    st.integers(min_value=min_x_width, max_value=100),
                )
            )
    elif type == "2d" or type == "depthwise":
        if type == "depthwise":
            filter_shape = draw(
                st.tuples(
                    st.integers(3, 5),
                    st.integers(3, 5),
                    st.integers(1, 3),
                )
            )
        else:
            filter_shape = draw(
                st.tuples(
                    st.integers(3, 5),
                    st.integers(3, 5),
                    st.integers(1, 3),
                    st.integers(1, 3),
                )
            )

        min_x_height = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
        min_x_width = filter_shape[1] + (filter_shape[1] - 1) * (dilations - 1)
        d_in = filter_shape[2]
        if data_format == "NHWC":
            x_shape = draw(
                st.tuples(
                    st.integers(1, 5),
                    st.integers(min_value=min_x_height, max_value=100),
                    st.integers(min_value=min_x_width, max_value=100),
                    st.integers(d_in, d_in),
                )
            )
        else:
            x_shape = draw(
                st.tuples(
                    st.integers(1, 5),
                    st.integers(d_in, d_in),
                    st.integers(min_value=min_x_height, max_value=100),
                    st.integers(min_value=min_x_width, max_value=100),
                )
            )

    else:
        filter_shape = draw(
            st.tuples(
                st.integers(3, 5),
                st.integers(3, 5),
                st.integers(3, 5),
                st.integers(1, 3),
                st.integers(1, 3),
            )
        )

        min_x_depth = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
        min_x_height = filter_shape[1] + (filter_shape[1] - 1) * (dilations - 1)
        min_x_width = filter_shape[2] + (filter_shape[2] - 1) * (dilations - 1)
        d_in = filter_shape[3]
        if data_format == "NDHWC":
            x_shape = draw(
                st.tuples(
                    st.integers(1, 5),
                    st.integers(min_value=min_x_depth, max_value=100),
                    st.integers(min_value=min_x_height, max_value=100),
                    st.integers(min_value=min_x_width, max_value=100),
                    st.integers(d_in, d_in),
                )
            )
        else:
            x_shape = draw(
                st.tuples(
                    st.integers(1, 5),
                    st.integers(d_in, d_in),
                    st.integers(min_value=min_x_depth, max_value=100),
                    st.integers(min_value=min_x_width, max_value=100),
                    st.integers(min_value=min_x_width, max_value=100),
                )
            )
    x = draw(helpers.array_values(dtype=dtype, shape=x_shape, min_value=0, max_value=1))
    filters = draw(
        helpers.array_values(dtype=dtype, shape=filter_shape, min_value=0, max_value=1)
    )
    return dtype, x, filters, dilations, data_format


# conv1d
@given(
    x_f_d_df=x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NWC", "NCW"]),
        type="1d",
    ),
    stride=st.integers(min_value=1, max_value=4),
    pad=st.sampled_from(["VALID", "SAME"]),
    num_positional_args=helpers.num_positional_args(fn_name="conv1d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv1d(
    *,
    data,
    x_f_d_df,
    stride,
    pad,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format = x_f_d_df
    dtype = [dtype] * 2
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
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
@pytest.mark.parametrize(
    "x_n_filters_n_pad_n_outshp_n_res",
    [
        (
            [[[0.0], [3.0], [0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
            "SAME",
            (1, 3, 1),
            [[[0.0], [3.0], [0.0]]],
        ),
        (
            [[[0.0], [3.0], [0.0]] for _ in range(5)],
            [[[0.0]], [[1.0]], [[0.0]]],
            "SAME",
            (5, 3, 1),
            [[[0.0], [3.0], [0.0]] for _ in range(5)],
        ),
        (
            [[[0.0], [3.0], [0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
            "VALID",
            (1, 5, 1),
            [[[0.0], [0.0], [3.0], [0.0], [0.0]]],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_conv1d_transpose(
    x_n_filters_n_pad_n_outshp_n_res, dtype, tensor_fn, device, call
):
    if call in [helpers.tf_call, helpers.tf_graph_call] and "cpu" in device:
        # tf conv1d transpose does not work when CUDA is installed, but array is on CPU
        pytest.skip()
    # smoke test
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not yet support conv1d_transpose
        pytest.skip()
    x, filters, padding, output_shape, true_res = x_n_filters_n_pad_n_outshp_n_res
    x = tensor_fn(x, dtype=dtype, device=device)
    filters = tensor_fn(filters, dtype=dtype, device=device)
    true_res = tensor_fn(true_res, dtype=dtype, device=device)
    ret = ivy.conv1d_transpose(x, filters, 1, padding, output_shape)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_res.shape
    # value test
    assert np.allclose(
        call(ivy.conv1d_transpose, x, filters, 1, padding, output_shape),
        ivy.to_numpy(true_res),
    )


# conv2d
@given(
    x_f_d_df=x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NHWC", "NCHW"]),
        type="2d",
    ),
    stride=st.integers(min_value=1, max_value=4),
    pad=st.sampled_from(["VALID", "SAME"]),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv2d(
    *,
    data,
    x_f_d_df,
    stride,
    pad,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format = x_f_d_df
    dtype = [dtype] * 2

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
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
    array_shape=helpers.lists(arg=st.integers(1, 5), min_size=3, max_size=3),
    filter_shape=st.integers(min_value=1, max_value=5),
    stride=st.integers(min_value=1, max_value=3),
    pad=st.sampled_from(["VALID", "SAME"]),
    output_shape=helpers.lists(arg=st.integers(1, 5), min_size=4, max_size=4),
    data_format=st.sampled_from(["NHWC", "NCHW"]),
    dilations=st.integers(min_value=1, max_value=5),
    dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_float_dtypes), length=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d_transpose"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv2d_transpose(
    *,
    data,
    array_shape,
    filter_shape,
    stride,
    pad,
    output_shape,
    data_format,
    dilations,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    # tf conv2d transpose does not work when CUDA is installed, but array is on CPU
    assume(not (fw == "tensorflow" and "cpu" in device))

    # numpy and jax do not yet support conv2d_transpose
    assume(not (fw in ["numpy", "jax"]))

    # not implemented for Half
    assume(not (fw == "torch" and ("16" in dtype[0] or "16" in dtype[1])))

    x = np.random.uniform(size=array_shape).astype(dtype[0])
    x = np.expand_dims(x, (-1))
    filters = np.random.uniform(size=(filter_shape, filter_shape, 1, 1)).astype(
        dtype[1]
    )

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv2d_transpose",
        device_=device,
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=tuple(output_shape),
        data_format=data_format,
        dilations=dilations,
    )


# depthwise_conv2d
@given(
    x_f_d_df=x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NHWC", "NCHW"]),
        type="depthwise",
    ),
    stride=st.integers(min_value=1, max_value=4),
    pad=st.sampled_from(["VALID", "SAME"]),
    num_positional_args=helpers.num_positional_args(fn_name="depthwise_conv2d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_depthwise_conv2d(
    *,
    x_f_d_df,
    stride,
    pad,
    num_positional_args,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    device,
):

    dtype, x, filters, dilations, data_format = x_f_d_df
    dtype = [dtype] * 2
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="depthwise_conv2d",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv3d
@given(
    x_f_d_df=x_and_filters(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        data_format=st.sampled_from(["NDHWC", "NCDHW"]),
        type="3d",
    ),
    stride=st.integers(min_value=1, max_value=4),
    pad=st.sampled_from(["VALID", "SAME"]),
    num_positional_args=helpers.num_positional_args(fn_name="conv3d"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv3d(
    *,
    data,
    x_f_d_df,
    stride,
    pad,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format = x_f_d_df
    dtype = [dtype] * 2

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv3d",
        x=np.asarray(x, dtype[0]),
        filters=np.asarray(filters, dtype[0]),
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv3d_transpose
@given(
    array_shape=helpers.lists(arg=st.integers(1, 5), min_size=4, max_size=4),
    filter_shape=st.integers(min_value=1, max_value=5),
    stride=st.integers(min_value=1, max_value=3),
    pad=st.sampled_from(["VALID", "SAME"]),
    output_shape=helpers.lists(arg=st.integers(1, 5), min_size=5, max_size=5),
    data_format=st.sampled_from(["NHWC", "NCHW"]),
    dilations=st.integers(min_value=1, max_value=5),
    dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_float_dtypes), length=2
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv3d_transpose"),
    data=st.data(),
)
@handle_cmd_line_args
def test_conv3d_transpose(
    *,
    data,
    array_shape,
    filter_shape,
    stride,
    pad,
    output_shape,
    data_format,
    dilations,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    # tf conv3d transpose does not work when CUDA is installed, but array is on CPU
    assume(not (fw == "tensorflow" and "cpu" in device))

    # numpy and jax do not yet support 3d transpose convolutions,
    # and mxnet only supports with CUDNN
    assume(not (fw in ["numpy", "jax"]))

    # mxnet only supports 3d transpose convolutions with CUDNN
    assume(not (fw == "mxnet" and "cpu" in device))

    # not implemented for half
    assume(not (fw == "torch" and ("16" in dtype[0] or "16" in dtype[1])))

    x = np.random.uniform(size=array_shape).astype(dtype[0])
    x = np.expand_dims(x, (-1))
    filters = np.random.uniform(
        size=(filter_shape, filter_shape, filter_shape, 1, 1)
    ).astype(dtype[1])

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv3d_transpose",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# LSTM #
# -----#

# lstm
@given(
    dtype_lstm=x_and_weight(
        dtypes=st.sampled_from(ivy_np.valid_float_dtypes),
        fn_name= "lstm_update",
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
    dtype, x_lstm, init_h, init_c, kernel, recurrent_kernel, bias, recurrent_bias = dtype_lstm
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
