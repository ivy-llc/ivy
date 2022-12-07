# global
import ivy
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _x_and_filters(
    draw,
    dtypes,
    data_format,
    padding,
    stride_min=1,
    stride_max=4,
    dilation_min=1,
    dilation_max=4,
    type: str = "2d",
    transpose=False,
    atrous=False,
):
    data_format = draw(data_format)
    dtype = draw(dtypes)
    padding = draw(padding)
    dilations = draw(helpers.ints(min_value=dilation_min, max_value=dilation_max))
    if transpose and atrous:
        stride = dilations
    else:
        stride = draw(helpers.ints(min_value=stride_min, max_value=stride_max))
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
                    st.integers(min_value=3, max_value=5),
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
                x_shape[0],
                ivy.deconv_length(x_w, stride, filter_shape[0], padding, dilations),
                d_in,
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
            output_shape_h = ivy.deconv_length(
                x_h, stride, filter_shape[0], padding, dilations
            )
            output_shape_w = ivy.deconv_length(
                x_w, stride, filter_shape[1], padding, dilations
            )
            output_shape = [x_shape[0], output_shape_h, output_shape_w, d_in]
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
            output_shape_d = ivy.deconv_length(
                x_d, stride, filter_shape[0], padding, dilations
            )
            output_shape_h = ivy.deconv_length(
                x_h, stride, filter_shape[1], padding, dilations
            )
            output_shape_w = ivy.deconv_length(
                x_w, stride, filter_shape[2], padding, dilations
            )
            output_shape = [output_shape_d, output_shape_h, output_shape_w]
    x = draw(
        helpers.array_values(dtype=dtype[0], shape=x_shape, min_value=0, max_value=1)
    )
    filters = draw(
        helpers.array_values(
            dtype=dtype[0], shape=filter_shape, min_value=0, max_value=1
        )
    )
    if not transpose:
        return dtype, x, filters, dilations, data_format, stride, padding
    return dtype, x, filters, dilations, data_format, stride, padding, output_shape


@handle_frontend_test(
    fn_tree="tensorflow.nn.atrous_conv2d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=1,
        type="2d",
    ),
)
def test_tensorflow_atrous_conv2d(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x,
        filters=filters,
        rate=dilations,
        padding=pad,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.atrous_conv2d_transpose",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=1,
        dilation_max=1,
        type="2d",
        transpose=True,
        atrous=True,
    ),
)
def test_tensorflow_atrous_conv2d_transpose(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    (
        input_dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        output_shape,
    ) = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x,
        filters=filters,
        output_shape=output_shape,
        rate=dilations,
        padding=pad,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv1d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=3,
        stride_max=4,
        type="1d",
    ),
)
def test_tensorflow_conv1d(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filters=filters,
        stride=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv1d_transpose",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=3,
        stride_max=4,
        dilation_max=1,
        type="1d",
        transpose=True,
    ),
)
def test_tensorflow_conv1d_transpose(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    (
        input_dtype,
        x,
        filters,
        dilations,
        data_format,
        stride,
        pad,
        output_shape,
    ) = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filters=filters,
        output_shape=output_shape,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    approximate=st.booleans(),
)
def test_tensorflow_gelu(
    *,
    dtype_and_x,
    approximate,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        features=x[0],
        approximate=approximate,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv2d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        type="2d",
    ),
)
def test_tensorflow_conv2d(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filters=filters,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilation,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv2d_transpose",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        type="2d",
        transpose=True,
    ),
)
def test_tensorflow_conv2d_transpose(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    (
        input_dtype,
        x,
        filters,
        dilation,
        data_format,
        stride,
        padding,
        output_shape,
    ) = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filters=filters,
        output_shape=output_shape,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=1 if not ivy.gpu_is_available() else dilation,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv3d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NDHWC"]),
        padding=st.sampled_from(["SAME"]),
        type="3d",
    ),
)
def test_tensorflow_conv3d(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    x = x[0]
    filters = filters[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filters=filters.reshape(
            filters.shape[:-2] + (x.shape[-1],) + (filters.shape[-1],)
        ),
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilation,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv3d_transpose",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NDHWC"]),
        padding=st.sampled_from(["SAME"]),
        type="3d",
        transpose=True,
    ),
)
def test_tensorflow_conv3d_transpose(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    (
        input_dtype,
        x,
        filters,
        dilation,
        data_format,
        stride,
        padding,
        output_shape,
    ) = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filters=filters,
        output_shape=output_shape,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=1 if not ivy.gpu_is_available() else dilation,
    )


# TODO: test with other dtypes
@handle_frontend_test(
    fn_tree="tensorflow.nn.batch_normalization",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        shape=(3, 5),
    ),
    mean=helpers.array_values(dtype=ivy.float16, shape=(3, 5), min_value=0),
    variance=helpers.array_values(dtype=ivy.float16, shape=(3, 5), min_value=0),
    offset=helpers.array_values(dtype=ivy.float16, shape=(3, 5)),
    scale=helpers.array_values(dtype=ivy.float16, shape=(3, 5)),
)
def test_tensorflow_batch_normalization(
    *,
    dtype_and_x,
    mean,
    variance,
    offset,
    scale,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        mean=mean[0],
        variance=variance[0],
        offset=offset[0],
        scale=scale[0],
        variance_epsilon=1e-7,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.dropout",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        shape=(3, 5),
    ),
    prob=helpers.array_values(dtype=ivy.float16, shape=(3, 5), min_value=0),
    scale=helpers.array_values(dtype=ivy.float16, shape=(3, 5), min_value=0),
)
def test_tensorflow_dropout(
    *,
    dtype_and_x,
    prob,
    scale,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[input_dtype] * 2,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        prob=prob,
        scale=scale,
    )


# silu
@handle_frontend_test(
    fn_tree="tensorflow.nn.silu",
    dtype_features=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=5,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    beta=st.one_of(
        helpers.floats(
            min_value=0,
            max_value=3,
        )
    ),
)
def test_tensorflow_silu(
    *,
    dtype_features,
    beta,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, features = dtype_features
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        features=features[0],
        beta=beta,
    )


# sigmoid_cross_entropy_with_logits
@handle_frontend_test(
    fn_tree="tensorflow.nn.sigmoid_cross_entropy_with_logits",
    dtype_labels_logits=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=0,
        max_value=1,
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_sigmoid_cross_entropy_with_logits(
    *,
    dtype_labels_logits,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, input_values = dtype_labels_logits
    labels, logits = input_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        labels=labels,
        logits=logits,
    )


# weighted_cross_entropy_with_logits
@handle_frontend_test(
    fn_tree="tensorflow.nn.weighted_cross_entropy_with_logits",
    dtype_labels_logits=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=0,
        max_value=1,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        shared_dtype=True,
    ),
    pos_weight=st.one_of(
        helpers.floats(
            min_value=0,
            max_value=3,
        )
    ),
)
def test_tensorflow_weighted_cross_entropy_with_logits(
    *,
    dtype_labels_logits,
    pos_weight,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, input_values = dtype_labels_logits
    labels, logits = input_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        labels=labels,
        logits=logits,
        pos_weight=pos_weight,
    )


# local_response_normalization
@handle_frontend_test(
    fn_tree="tensorflow.nn.local_response_normalization",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-20,
        max_value=20,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
        large_abs_safety_factor=1.5,
        small_abs_safety_factor=1.5,
    ),
    depth_radius=st.integers(min_value=1, max_value=7),
    bias=st.floats(min_value=0.1, max_value=30),
    alpha=st.floats(min_value=0.1, max_value=20),
    beta=st.floats(min_value=0.1, max_value=5),
)
def test_tensorflow_local_response_normalization(
    *,
    dtype_and_x,
    depth_radius,
    bias,
    alpha,
    beta,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    input = x[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-3,
        atol=1e-3,
        input=input,
        depth_radius=depth_radius,
        bias=bias,
        alpha=alpha,
        beta=beta,
    )


@st.composite
def df(draw, data_format):
    data_format = draw(data_format)
    return data_format


# max_pool1d
@handle_frontend_test(
    fn_tree="tensorflow.nn.max_pool1d",
    data_format=df(data_format=st.sampled_from(["NWC"])),
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
)
def test_tensorflow_max_pool1d(
    *,
    x_k_s_p,
    data_format,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, ksize, strides, padding = x_k_s_p
    data_format = data_format
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )
