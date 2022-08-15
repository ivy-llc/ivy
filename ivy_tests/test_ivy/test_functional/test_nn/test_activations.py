"""Collection of tests for unified neural network activation functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# relu
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="relu"),
    data=st.data(),
)
@handle_cmd_line_args
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="leaky_relu"),
    alpha=st.floats(width=16),
    data=st.data(),
)
@handle_cmd_line_args
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
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    approximate=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="gelu"),
    data=st.data(),
)
@handle_cmd_line_args
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
        atol_=1e-4,
        rtol_=1e-4,
        x=np.asarray(x, dtype=dtype),
        approximate=approximate,
    )


# sigmoid
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="sigmoid"),
    data=st.data(),
)
@handle_cmd_line_args
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
        x=np.asarray(x, dtype=dtype),
    )


# softmax
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, min_num_dims=1
    ),
    axis=st.integers(-1, 0),
    num_positional_args=helpers.num_positional_args(fn_name="softmax"),
    data=st.data(),
)
@handle_cmd_line_args
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
        x=np.asarray(x, dtype=dtype),
        axis=axis,
    )


# softplus
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, min_num_dims=1
    ),
    num_positional_args=helpers.num_positional_args(fn_name="softplus"),
    data=st.data(),
)
@handle_cmd_line_args
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
        x=np.asarray(x, dtype=dtype),
    )
