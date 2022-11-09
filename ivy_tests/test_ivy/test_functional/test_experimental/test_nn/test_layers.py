# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
    num_positional_args=helpers.num_positional_args(fn_name="max_pool2d"),
)
def test_max_pool2d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="max_pool2d",
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="jax",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_cmd_line_args
@given(
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
    num_positional_args=helpers.num_positional_args(fn_name="max_pool1d"),
)
def test_max_pool1d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="max_pool1d",
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="jax",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_cmd_line_args
@given(
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
    num_positional_args=helpers.num_positional_args(fn_name="max_pool3d"),
)
def test_max_pool3d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="max_pool3d",
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="jax",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_cmd_line_args
@given(
    x_k_s_p=helpers.arrays_for_pooling(min_dims=5, max_dims=5, min_side=1, max_side=4),
    num_positional_args=helpers.num_positional_args(fn_name="avg_pool3d"),
)
def test_avg_pool3d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="avg_pool3d",
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="jax",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@handle_cmd_line_args
@given(
    x_k_s_p=helpers.arrays_for_pooling(min_dims=4, max_dims=4, min_side=1, max_side=4),
    num_positional_args=helpers.num_positional_args(fn_name="avg_pool2d"),
)
def test_avg_pool2d(
    *,
    x_k_s_p,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, kernel, stride, pad = x_k_s_p
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="avg_pool2d",
        rtol_=1e-2,
        atol_=1e-2,
        ground_truth_backend="tensorflow",
        x=x[0],
        kernel=kernel,
        strides=stride,
        padding=pad,
    )


@st.composite
def valid_dct(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
            shared_dtype=True,
        )
    )
    dims_len = len(x[0].shape)
    n = draw(st.sampled_from([None, "int"]))
    axis = draw(helpers.ints(min_value=-dims_len, max_value=dims_len))
    norm = draw(st.sampled_from([None, "ortho"]))
    type = draw(helpers.ints(min_value=1, max_value=4))
    if n == "int":
        n = draw(helpers.ints(min_value=1, max_value=20))
        if n <= 1 and type == 1:
            n = 2
    if norm == "ortho" and type == 1:
        norm = None
    return dtype, x, type, n, axis, norm


@handle_cmd_line_args
@given(
    dtype_x_and_args=valid_dct(),
    num_positional_args=helpers.num_positional_args(fn_name="dct"),
)
def test_dct(
    dtype_x_and_args,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):  
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="dct",
        x=x[0],
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        rtol_=1e-3,
        atol_=1e-1,
    )
