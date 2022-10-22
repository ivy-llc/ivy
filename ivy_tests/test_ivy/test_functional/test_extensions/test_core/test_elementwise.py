# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Helpers #
# ------- #


# sinc
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="sinc"),
)
def test_sinc(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sinc",
        x=x[0],
    )


# lcm
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="lcm"),
)
def test_lcm(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="lcm",
        test_gradients=True,
        x1=x[0],
        x2=x[1],
    )


# fmod
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="fmod"),
)
def test_fmod(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="fmod",
        x1=x[0],
        x2=x[0],
    )


# fmax
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        allow_nan=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="fmax"),
)
def test_fmax(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="fmax",
        x1=x[0],
        x2=x[0],
    )
