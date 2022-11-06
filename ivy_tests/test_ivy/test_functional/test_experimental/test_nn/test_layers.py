# global
from hypothesis import given

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
