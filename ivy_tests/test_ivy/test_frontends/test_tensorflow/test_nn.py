# global
import ivy
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.ivy.layers import _deconv_length
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
from ivy_tests.test_ivy.test_functional.test_nn.test_layers import _dropout_helper


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

    # Infer type from data_format if it is passed as None
    if type is None:
        type_data_format_mapping = {
            "1d": ["NWC", "NCW"],
            "2d": ["NHWC", "NCHW"],
            "3d": ["NDHWC", "NCDHW"],
        }
        type = [
            typ
            for typ in type_data_format_mapping
            if data_format in type_data_format_mapping[typ]
        ][0]

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
                _deconv_length(x_w, stride, filter_shape[0], padding, dilations),
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
                    helpers.ints(min_value=1, max_value=1),
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
                    helpers.ints(min_value=min_x_height, max_value=100),
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
    test_with_out=st.just(False),
)
def test_tensorflow_atrous_conv2d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_atrous_conv2d_transpose(
    *,
    x_f_d_df,
    frontend,
    test_flags,
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
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_conv1d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_conv1d_transpose(
    *,
    x_f_d_df,
    frontend,
    test_flags,
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
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_gelu(
    *,
    dtype_and_x,
    approximate,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_conv2d_transpose(
    *,
    x_f_d_df,
    frontend,
    test_flags,
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
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_conv3d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    x = x[0]
    filters = filters[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_conv3d_transpose(
    *,
    x_f_d_df,
    frontend,
    test_flags,
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
        frontend=frontend,
        test_flags=test_flags,
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
    fn_tree="tensorflow.nn.depthwise_conv2d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        type="depthwise",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_depthwise_conv2d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    stride = 1 if dilation > 1 else stride
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filter=filters,
        strides=[1, stride, stride, 1],
        padding=padding,
        data_format=data_format,
        dilations=[dilation, dilation],
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
    test_with_out=st.just(False),
)
def test_tensorflow_batch_normalization(
    *,
    dtype_and_x,
    mean,
    variance,
    offset,
    scale,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    dtype_x_noiseshape=_dropout_helper(),
    rate=helpers.floats(min_value=0, max_value=0.9),
    seed=helpers.ints(min_value=0, max_value=100),
    test_with_out=st.just(False),
)
def test_tensorflow_dropout(
    *,
    dtype_x_noiseshape,
    rate,
    seed,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    (x_dtype, x), noise_shape = dtype_x_noiseshape
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x[0],
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    frontend_ret = helpers.flatten_and_to_np(ret=frontend_ret)
    for u, v, w in zip(ret, frontend_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape


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
    test_with_out=st.just(False),
)
def test_tensorflow_silu(
    *,
    dtype_features,
    beta,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, features = dtype_features
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_sigmoid_cross_entropy_with_logits(
    *,
    dtype_labels_logits,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, input_values = dtype_labels_logits
    labels, logits = input_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_weighted_cross_entropy_with_logits(
    *,
    dtype_labels_logits,
    pos_weight,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, input_values = dtype_labels_logits
    labels, logits = input_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_local_response_normalization(
    *,
    dtype_and_x,
    depth_radius,
    bias,
    alpha,
    beta,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    input = x[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_tensorflow_max_pool1d(
    *,
    x_k_s_p,
    data_format,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, ksize, strides, padding = x_k_s_p
    data_format = data_format
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )


# moments
@handle_frontend_test(
    fn_tree="tensorflow.nn.moments",
    dtype_x_axis=statistical_dtype_values(function="mean"),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_moments(
    *,
    dtype_x_axis,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        x=x[0],
        axes=axis,
        keepdims=keepdims,
    )


@st.composite
def _generate_bias_data(draw):
    data_format = draw(st.sampled_from(["NC...", "N...C", None]))
    channel_dim = 1 if data_format == "NC..." else -1
    dtype, value, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_num_dims=3,
            ret_shape=True,
        )
    )
    channel_size = shape[channel_dim]
    bias = draw(helpers.array_values(dtype=dtype[0], shape=(channel_size,)))
    return data_format, dtype, value, bias


@handle_frontend_test(
    fn_tree="tensorflow.nn.bias_add",
    data=_generate_bias_data(),
    test_with_out=st.just(False),
)
def test_tensorflow_bias_add(
    *,
    data,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    data_format, dtype, value, bias = data
    helpers.test_frontend_function(
        input_dtypes=dtype * 2,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=value[0],
        bias=bias,
        data_format=data_format,
    )


# convolution
@handle_frontend_test(
    fn_tree="tensorflow.nn.convolution",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NWC", "NHWC", "NDHWC"]),
        padding=st.sampled_from(["SAME", "VALID"]),
        # Tensorflow backprop doesn't support dilations more than 1 on CPU
        dilation_min=1,
        dilation_max=1,
        type=None,
        transpose=False,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_convolution(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filters=filters,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilation,
    )


# relu
@handle_frontend_test(
    fn_tree="tensorflow.nn.relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_value=-20,
        max_value=20,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_relu(
    *,
    dtype_and_x,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        features=x[0],
    )


# softmax
@handle_frontend_test(
    fn_tree="tensorflow.nn.softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=4,
        max_axes_size=3,
        force_int_axis=True,
        valid_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_softmax(
    *,
    dtype_x_and_axis,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        logits=x[0],
        axis=axis,
    )


# embedding_lookup
@handle_frontend_test(
    fn_tree="tensorflow.nn.embedding_lookup",
    dtypes_indices_weights=helpers.embedding_helper(),
    max_norm=st.floats(min_value=0, max_value=5, exclude_min=True),
)
def test_tensorflow_embedding_lookup(
    *,
    dtypes_indices_weights,
    max_norm,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, indices, weight, _ = dtypes_indices_weights
    dtypes.reverse()
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        params=weight,
        ids=indices,
        max_norm=max_norm,
    )
