"""Collection of tests for unified neural network activation functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# relu
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(fn_name="relu"),
)
def test_relu(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    container,
    instance_method,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        num_positional_args=num_positional_args,
        container_flags=container,
        instance_method=instance_method,
        fn_name="relu",
        x=np.asarray(x, dtype=dtype),
    )


# leaky_relu
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(fn_name="leaky_relu"),
    alpha=st.floats(width=16),
)
def test_leaky_relu(
    *,
    dtype_and_x,
    alpha,
    as_variable,
    with_out,
    num_positional_args,
    container,
    instance_method,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        num_positional_args=num_positional_args,
        container_flags=container,
        instance_method=instance_method,
        fn_name="leaky_relu",
        rtol_=1e-4,
        x=np.asarray(x, dtype=dtype),
        alpha=alpha,
    )


# gelu
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=2.0,
    ),
    approximate=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="gelu"),
)
def test_gelu(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    approximate,
    num_positional_args,
    container,
    instance_method,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        num_positional_args=num_positional_args,
        container_flags=container,
        instance_method=instance_method,
        fn_name="gelu",
        atol_=1e-2,
        rtol_=1e-2,
        x=np.asarray(x, dtype=dtype),
        approximate=approximate,
    )


# sigmoid
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(fn_name="sigmoid"),
)
def test_sigmoid(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    container,
    instance_method,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        num_positional_args=num_positional_args,
        container_flags=container,
        instance_method=instance_method,
        fn_name="sigmoid",
        rtol_=1e-2,
        atol_=1e-2,
        x=np.asarray(x, dtype=dtype),
    )


# softmax
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(fn_name="softmax"),
)
def test_softmax(
    *,
    dtype_and_x,
    as_variable,
    axis,
    with_out,
    num_positional_args,
    container,
    instance_method,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        num_positional_args=num_positional_args,
        container_flags=container,
        instance_method=instance_method,
        fn_name="softmax",
        rtol_=1e-02,
        atol_=1e-02,
        x=np.asarray(x, dtype=dtype),
        axis=axis,
    )


# softplus
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1
    ),
    num_positional_args=helpers.num_positional_args(fn_name="softplus"),
)
def test_softplus(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    container,
    instance_method,
    native_array,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        num_positional_args=num_positional_args,
        container_flags=container,
        instance_method=instance_method,
        fn_name="softplus",
        rtol_=1e-02,
        atol_=1e-02,
        x=np.asarray(x, dtype=dtype),
    )
