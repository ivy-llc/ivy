# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


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
    x = draw(helpers.array_values(dtype=dtype, shape=x_shape, min_value=0, max_value=1))
    filters = draw(
        helpers.array_values(dtype=dtype, shape=filter_shape, min_value=0, max_value=1)
    )
    if not transpose:
        return dtype, x, filters, dilations, data_format, stride, padding
    return dtype, x, filters, dilations, data_format, stride, padding, output_shape


@handle_cmd_line_args
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(
            tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_tf.valid_float_dtypes)
                )
            )
        ),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=1,
        type="2d",
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.atrous_conv2d"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_tensorflow_atrous_conv2d(
    x_f_d_df, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    input_dtype = [input_dtype] * 2
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="nn.atrous_conv2d",
        value=np.asarray(x, dtype=input_dtype[0]),
        filters=np.asarray(filters, dtype=input_dtype[1]),
        rate=dilations,
        padding=pad,
    )


@handle_cmd_line_args
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(
            tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_tf.valid_float_dtypes)
                )
            )
        ),
        data_format=st.sampled_from(["NHWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=1,
        stride_max=1,
        dilation_max=1,
        type="2d",
        transpose=True,
        atrous=True,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.atrous_conv2d_transpose"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_tensorflow_atrous_conv2d_transpose(
    x_f_d_df, as_variable, num_positional_args, native_array, fw
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
    input_dtype = [input_dtype] * 2
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="nn.atrous_conv2d_transpose",
        value=np.asarray(x, dtype=input_dtype[0]),
        filters=np.asarray(filters, dtype=input_dtype[1]),
        output_shape=output_shape,
        rate=dilations,
        padding=pad,
    )


@handle_cmd_line_args
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(
            tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_tf.valid_float_dtypes)
                )
            )
        ),
        data_format=st.sampled_from(["NWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=3,
        stride_max=4,
        type="1d",
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.conv1d"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_tensorflow_conv1d(
    x_f_d_df, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    input_dtype = [input_dtype] * 2
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="nn.conv1d",
        input=np.asarray(x, dtype=input_dtype[0]),
        filters=np.asarray(filters, dtype=input_dtype[1]),
        stride=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


@handle_cmd_line_args
@given(
    x_f_d_df=_x_and_filters(
        dtypes=st.sampled_from(
            tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_tf.valid_float_dtypes)
                )
            )
        ),
        data_format=st.sampled_from(["NWC"]),
        padding=st.sampled_from(["VALID", "SAME"]),
        stride_min=3,
        stride_max=4,
        dilation_max=1,
        type="1d",
        transpose=True,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.conv1d_transpose"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_tensorflow_conv1d_transpose(
    x_f_d_df, as_variable, num_positional_args, native_array, fw
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
    input_dtype = [input_dtype] * 2
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="nn.conv1d_transpose",
        input=np.asarray(x, dtype=input_dtype[0]),
        filters=np.asarray(filters, dtype=input_dtype[1]),
        output_shape=output_shape,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )
