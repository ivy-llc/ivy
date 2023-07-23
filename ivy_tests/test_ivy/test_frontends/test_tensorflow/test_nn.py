# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.ivy.layers import _deconv_length
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
from ivy_tests.test_ivy.test_functional.test_nn.test_layers import (
    _assume_tf_dilation_gt_1,
)


@handle_frontend_test(
    fn_tree="tensorflow.nn.leaky_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=25,
        small_abs_safety_factor=25,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
    alpha=helpers.floats(
        min_value=0,
        max_value=1,
        large_abs_safety_factor=25,
        small_abs_safety_factor=25,
        safety_factor_scale="log",
    ),
)
def test_tensorflow_leaky_relu(
    *,
    dtype_and_x,
    alpha,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x = dtype_and_x
    return helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        features=x[0],
        alpha=alpha,
    )


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
    if type is not None:
        if "1" in type:
            dim = 1
        elif "2" in type:
            dim = 2
        elif "3" in type:
            dim = 3
        elif type in ["depthwise", "separable"]:
            dim = 2
    else:
        dim = len(data_format) - 2
    if padding == "EXPLICIT":
        padding = draw(
            helpers.lists(
                x=st.integers(min_value=0, max_value=2),
                min_size=dim * 2,
                max_size=dim * 2,
            )
        )
        if data_format.find("C") == 1:
            padding = [1, 1, 1, 1] + padding
        else:
            padding = [0, 0] + padding + [0, 0]
    if atrous:
        dilations = draw(st.integers(dilation_min, dilation_max))
    else:
        dilations = draw(
            st.one_of(
                st.integers(dilation_min, dilation_max),
                st.lists(
                    st.integers(dilation_min, dilation_max), min_size=dim, max_size=dim
                ),
            )
        )
    fdilations = [dilations] * dim if isinstance(dilations, int) else dilations
    if atrous:
        stride = 1
    elif type in ["depthwise", "separable"]:
        # if any value in dilations is greater than 1, tensorflow implements
        # depthwise_covn2d as an atrous depthwise convolution, in which case all values
        # in strides must be equal to 1.
        if any(x > 1 for x in fdilations):
            stride = 1
        else:
            stride = draw(st.integers(stride_min, stride_max))
    else:
        stride = draw(
            st.one_of(
                st.integers(stride_min, stride_max),
                st.lists(
                    st.integers(stride_min, stride_max), min_size=dim, max_size=dim
                ),
            )
        )
    fstride = [stride] * dim if isinstance(stride, int) else stride
    if dim == 1:
        if not transpose:
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=1, max_value=3),
                    helpers.ints(min_value=1, max_value=3),
                )
            )
            min_x_width = filter_shape[0] + (filter_shape[0] - 1) * (fdilations[0] - 1)
        else:
            filter_shape = draw(
                st.tuples(
                    helpers.ints(min_value=3, max_value=5),
                    helpers.ints(min_value=1, max_value=3),
                    helpers.ints(min_value=1, max_value=3),
                )
            )
            min_x_width = 1
        if transpose:
            d_in = filter_shape[2]
        else:
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
                _deconv_length(
                    x_w, fstride[0], filter_shape[0], padding, fdilations[0]
                ),
                filter_shape[1],
            ]
    elif dim == 2:
        min_x_height = 1
        min_x_width = 1
        filter_shape = draw(
            st.tuples(
                helpers.ints(min_value=3, max_value=5),
                helpers.ints(min_value=3, max_value=5),
                helpers.ints(min_value=1, max_value=3),
                helpers.ints(min_value=1, max_value=3),
            )
        )
        if not transpose:
            min_x_height = filter_shape[0] + (filter_shape[0] - 1) * (fdilations[0] - 1)
            min_x_width = filter_shape[1] + (filter_shape[1] - 1) * (fdilations[1] - 1)
        if transpose:
            d_in = filter_shape[3]
        else:
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
                x_h, fstride[0], filter_shape[0], padding, fdilations[0]
            )
            output_shape_w = _deconv_length(
                x_w, fstride[1], filter_shape[1], padding, fdilations[1]
            )
            output_shape = [x_shape[0], output_shape_h, output_shape_w, filter_shape[2]]
    elif dim == 3:
        filter_shape = draw(
            st.tuples(
                helpers.ints(min_value=3, max_value=5),
                helpers.ints(min_value=3, max_value=5),
                helpers.ints(min_value=3, max_value=5),
                helpers.ints(min_value=1, max_value=3),
                helpers.ints(min_value=1, max_value=3),
            )
        )
        if not transpose:
            min_x_depth = filter_shape[0] + (filter_shape[0] - 1) * (fdilations[0] - 1)
            min_x_height = filter_shape[1] + (filter_shape[1] - 1) * (fdilations[1] - 1)
            min_x_width = filter_shape[2] + (filter_shape[2] - 1) * (fdilations[2] - 1)
        else:
            min_x_depth = 1
            min_x_height = 1
            min_x_width = 1
        if transpose:
            d_in = filter_shape[4]
        else:
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
                x_d, fstride[0], filter_shape[0], padding, fdilations[0]
            )
            output_shape_h = _deconv_length(
                x_h, fstride[1], filter_shape[1], padding, fdilations[1]
            )
            output_shape_w = _deconv_length(
                x_w, fstride[2], filter_shape[2], padding, fdilations[2]
            )
            output_shape = [
                x_shape[0],
                output_shape_d,
                output_shape_h,
                output_shape_w,
                filter_shape[3],
            ]
    x = draw(
        helpers.array_values(dtype=dtype[0], shape=x_shape, min_value=0, max_value=1)
    )
    filters = draw(
        helpers.array_values(
            dtype=dtype[0], shape=filter_shape, min_value=0, max_value=1
        )
    )
    if type == "separable":
        p_filter_shape = (
            1,
            1,
            filter_shape[-1] * filter_shape[-2],
            draw(helpers.ints(min_value=1, max_value=3)),
        )
        p_filters = draw(
            helpers.array_values(
                dtype=dtype[0], shape=p_filter_shape, min_value=0, max_value=1
            )
        )
        filters = [filters, p_filters]
    if type in ["depthwise", "separable"]:
        stride = [1, stride, stride, 1]
        if isinstance(dilations, int):
            dilations = [dilations] * dim
    elif not atrous and type is not None:
        if transpose:
            if isinstance(stride, int):
                stride = [stride]
            else:
                if draw(st.booleans()):
                    stride = [1, *stride, 1]
            if isinstance(dilations, int):
                dilations = [dilations]
            else:
                if draw(st.booleans()):
                    dilations = [1, *dilations, 1]
        else:
            if dim != 3:
                if isinstance(stride, int):
                    stride = [stride]
                else:
                    if draw(st.booleans()):
                        stride = [1, *stride, 1]
                if isinstance(dilations, int):
                    dilations = [dilations]
                else:
                    if draw(st.booleans()):
                        dilations = [1, *dilations, 1]
            else:
                if isinstance(stride, int):
                    stride = [stride] * dim
                stride = [1, *stride, 1]
                if isinstance(dilations, int):
                    dilations = [dilations] * dim
                dilations = [1, *dilations, 1]
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
        atrous=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_atrous_conv2d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
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
    _assume_tf_dilation_gt_1("tensorflow", on_device, dilations)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
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
    _assume_tf_dilation_gt_1("tensorflow", on_device, dilations)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    fn_tree="tensorflow.nn.gelu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_value=1e04,
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
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
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
    _assume_tf_dilation_gt_1("tensorflow", on_device, dilation)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
        dilations=dilation,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.conv3d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NDHWC"]),
        padding=st.sampled_from(["SAME"]),
        type="3d",
        dilation_max=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_conv3d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
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
    _assume_tf_dilation_gt_1("tensorflow", on_device, dilation)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
        dilations=dilation,
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
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        filter=filters,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilation,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.separable_conv2d",
    x_f_d_df=_x_and_filters(
        dtypes=helpers.get_dtypes("float", full=False),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        type="separable",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_separable_conv2d(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        depthwise_filter=filters[0],
        pointwise_filter=filters[1],
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilation,
    )


@st.composite
def _batch_normalization_helper(draw):
    shape1, shape2, shape3, shape4 = draw(helpers.mutually_broadcastable_shapes(4))
    shape = helpers.broadcast_shapes(shape1, shape2, shape3, shape4)
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            shape=shape,
            max_value=999,
            min_value=-1001,
        )
    )

    _, mean = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape1,
            min_value=-1001,
            max_value=999,
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape2,
            min_value=0,
            max_value=999,
        )
    )
    _, offset = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape3,
            min_value=-1001,
            max_value=999,
        )
    )
    _, scale = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=shape4,
            min_value=-1001,
            max_value=999,
        )
    )

    return x_dtype, x[0], mean[0], variance[0], offset[0], scale[0]


@handle_frontend_test(
    fn_tree="tensorflow.nn.batch_normalization",
    data=_batch_normalization_helper(),
    eps=helpers.floats(min_value=1e-5, max_value=0.1),
)
def test_tensorflow_batch_normalization(
    *,
    data,
    eps,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    x_dtype, x, mean, variance, offset, scale = data
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        rtol=1e-2,
        atol=1e-2,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        mean=mean,
        variance=variance,
        offset=offset,
        scale=scale,
        variance_epsilon=eps,
    )


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
    seed = draw(helpers.ints(min_value=0, max_value=100))
    rate = draw(helpers.floats(min_value=0, max_value=0.9))

    return (
        dtype_and_x,
        noise_shape,
        seed,
        rate,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.dropout",
    dtype_x_noiseshape=_dropout_helper(),
)
def test_tensorflow_dropout(
    *,
    dtype_x_noiseshape,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    (x_dtype, x), noise_shape, seed, rate = dtype_x_noiseshape
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
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
    beta=helpers.floats(
        min_value=0,
        max_value=3,
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
    backend_fw,
    on_device,
):
    input_dtype, features = dtype_features
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
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
    backend_fw,
    on_device,
):
    input_dtype, input_values = dtype_labels_logits
    labels, logits = input_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    input_dtype, input_values = dtype_labels_logits
    labels, logits = input_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
        large_abs_safety_factor=25,
        small_abs_safety_factor=25,
    ),
    depth_radius=st.integers(min_value=1, max_value=5),
    bias=st.floats(min_value=0.1, max_value=1.5),
    alpha=st.floats(min_value=0.1, max_value=1.5),
    beta=st.floats(min_value=0.1, max_value=1.5),
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
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        input=x[0],
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
    backend_fw,
    on_device,
):
    input_dtype, x, ksize, strides, padding = x_k_s_p
    data_format = data_format
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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


# max_pool2d
@handle_frontend_test(
    fn_tree="tensorflow.nn.max_pool2d",
    data_format=df(data_format=st.sampled_from(["NHWC"])),
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
    test_with_out=st.just(False),
)
def test_tensorflow_max_pool2d(
    *,
    x_k_s_p,
    data_format,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, ksize, strides, padding = x_k_s_p
    data_format = data_format
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    dtype_x_axis=_statistical_dtype_values(function="mean"),
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
    backend_fw,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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


# Normalize Moments
@st.composite
def _normalize_moments_helper(draw):
    shape1, shape2, shape3 = draw(helpers.mutually_broadcastable_shapes(3))
    counts_dtype, counts = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            max_value=999,
            min_value=-1001,
            max_num_dims=1,
            max_dim_size=1,
            min_dim_size=1,
        )
    )
    _, mean = draw(
        helpers.dtype_and_values(
            available_dtypes=counts_dtype,
            shape=shape1,
            min_value=1,
            max_num_dims=1,
            max_dim_size=1,
            min_dim_size=1,
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            available_dtypes=counts_dtype,
            shape=shape2,
            min_value=1,
            max_num_dims=1,
            max_dim_size=1,
            min_dim_size=1,
        )
    )
    _, shift = draw(
        helpers.dtype_and_values(
            available_dtypes=counts_dtype,
            shape=shape3,
            min_value=1,
            max_num_dims=1,
            max_dim_size=1,
            min_dim_size=1,
        )
    )

    return counts_dtype, counts[0], mean[0], variance[0], shift[0]


@handle_frontend_test(
    fn_tree="tensorflow.nn.normalize_moments",
    data=_normalize_moments_helper(),
)
def test_tensorflow_normalize_moments(
    *,
    data,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    counts_dtype, counts, mean, variance, shift = data
    helpers.test_frontend_function(
        input_dtypes=counts_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        counts=counts,
        mean_ss=mean,
        variance_ss=variance,
        shift=shift,
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
    backend_fw,
    on_device,
):
    data_format, dtype, value, bias = data
    helpers.test_frontend_function(
        input_dtypes=dtype * 2,
        backend_to_test=backend_fw,
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
        dilation_max=1,
        type=None,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_convolution(
    *,
    x_f_d_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, filters, dilation, data_format, stride, padding = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        features=x[0],
    )


# relu6
@handle_frontend_test(
    fn_tree="tensorflow.nn.relu6",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_value=-20,
        max_value=20,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_relu6(
    *,
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
        min_num_dims=1,
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
    backend_fw,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    max_norm=st.floats(min_value=0.1, max_value=5, exclude_min=True),
)
def test_tensorflow_embedding_lookup(
    *,
    dtypes_indices_weights,
    max_norm,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    dtypes, indices, weight, _ = dtypes_indices_weights
    dtypes.reverse()
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        params=weight,
        ids=indices,
        max_norm=max_norm,
        atol=1e-4,
    )


# crelu
@handle_frontend_test(
    fn_tree="tensorflow.nn.crelu",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=4,
        max_axes_size=3,
        force_int_axis=True,
        valid_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_crelu(
    *,
    dtype_x_and_axis,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        features=x[0],
        axis=axis,
    )


@st.composite
def _average_pool_args(draw):
    dims = draw(st.integers(min_value=1, max_value=3))
    data_formats = ["NWC", "NHWC", "NDHWC"]
    data_format = data_formats[dims - 1]
    return (
        draw(
            helpers.arrays_for_pooling(
                min_dims=dims + 2, max_dims=dims + 2, min_side=1, max_side=4
            )
        ),
        data_format,
    )


# average_pool
@handle_frontend_test(
    fn_tree="tensorflow.nn.avg_pool",
    x_k_s_p_df=_average_pool_args(),
    test_with_out=st.just(False),
)
def test_tensorflow_avg_pool(
    *,
    x_k_s_p_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    (input_dtype, x, ksize, strides, padding), data_format = x_k_s_p_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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


@handle_frontend_test(
    fn_tree="tensorflow.nn.avg_pool3d",
    x_k_s_p_df=helpers.arrays_for_pooling(
        min_dims=5, max_dims=5, min_side=1, max_side=4
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_avg_pool3d(
    *,
    x_k_s_p_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, ksize, strides, padding = x_k_s_p_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        ksize=ksize,
        strides=strides,
        padding=padding,
    )


# test_avg_pool1d
@handle_frontend_test(
    fn_tree="tensorflow.nn.avg_pool1d",
    x_k_s_p_df=helpers.arrays_for_pooling(
        min_dims=3, max_dims=3, min_side=1, max_side=4
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_avg_pool1d(
    *,
    x_k_s_p_df,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    (input_dtype, x, ksize, strides, padding) = x_k_s_p_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        ksize=ksize,
        strides=strides,
        padding=padding,
    )


@st.composite
def _pool_args(draw):
    dims = draw(st.integers(min_value=3, max_value=5))
    data_formats = {3: "NWC", 4: "NHWC", 5: "NDHWC"}
    data_format = data_formats[dims]
    pooling_type = draw(st.one_of(st.just("AVG"), st.just("MAX")))
    return (
        draw(
            helpers.arrays_for_pooling(
                min_dims=dims,
                max_dims=dims,
                min_side=1,
                max_side=4,
                return_dilation=True,
            )
        ),
        data_format,
        pooling_type,
        dims,
    )


# pool
@handle_frontend_test(
    fn_tree="tensorflow.nn.pool",
    x_k_s_p_df=_pool_args(),
    test_with_out=st.just(False),
)
def test_tensorflow_pool(
    *,
    x_k_s_p_df,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    (
        (input_dtype, x, ksize, strides, padding, dilation),
        data_format,
        pooling_type,
        num_dims,
    ) = x_k_s_p_df
    if num_dims == 3:
        strides = (strides[0],)
    elif num_dims == 4:
        strides = (strides[0], strides[0])
    elif num_dims == 5:
        strides = (strides[0], strides[0], strides[0])
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        window_shape=ksize,
        pooling_type=pooling_type,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilation,
    )


# sufficient_statistics
@st.composite
def _axes_value(draw):
    s = draw(
        helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=5,
        )
    )
    dtype_and_x = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            shape=s,
            valid_axis=True,
            force_tuple_axis=True,
        )
    )
    return dtype_and_x


@handle_frontend_test(
    fn_tree="tensorflow.nn.sufficient_statistics",
    dtypes_x_axes_shift=_axes_value(),
    sh=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float"), shape=()),
    keepdims=st.booleans(),
)
def test_tensorflow_sufficient_statistics(
    *,
    dtypes_x_axes_shift,
    sh,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtypes, x, a = dtypes_x_axes_shift
    return helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axes=a,
        shift=sh[1][0],
        keepdims=keepdims,
        name=None,
    )


@handle_frontend_test(
    fn_tree="tensorflow.nn.log_poisson_loss",
    dtype_target_log_inputs=helpers.dtype_and_values(
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
    compute_full_loss=st.sampled_from([True, False]),
    test_with_out=st.just(False),
)
def test_log_poisson_loss(
    *,
    dtype_target_log_inputs,
    compute_full_loss,
    test_flags,
    frontend,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, input_values = dtype_target_log_inputs
    targets, log_input = input_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        targets=targets,
        log_input=log_input,
        compute_full_loss=compute_full_loss,
        atol=1e-2,
    )


# ctc_unique_labels
@handle_frontend_test(
    fn_tree="tensorflow.nn.ctc_unique_labels",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=["int64", "int32"],
        min_value=1,
        max_value=100,
        min_dim_size=1,
        max_dim_size=10,
        min_num_dims=2,
        max_num_dims=2,
    ),
    test_with_out=st.just([False]),
)
def test_tensorflow_ctc_unique_labels(
    *,
    dtype_x,
    frontend,
    fn_tree,
    test_flags,
    on_device,
    backend_fw,
):
    dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        labels=x[0],
    )


# weighted moments
@handle_frontend_test(
    fn_tree="tensorflow.nn.weighted_moments",
    dtype_and_x_and_axis=_statistical_dtype_values(function="mean"),
    dtype_and_fw=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_value=0.00001,
    ),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_weighted_moments(
    *,
    dtype_and_x_and_axis,
    dtype_and_fw,
    keepdims,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_and_x_and_axis
    fw_dtype, fw = dtype_and_fw
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        x=x[0],
        axes=axis,
        frequency_weights=fw[0],
        keepdims=keepdims,
    )
