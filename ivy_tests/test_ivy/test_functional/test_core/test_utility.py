"""Collection of tests for utility functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# all
@handle_cmd_line_args()
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=ivy_np.valid_int_dtypes,
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_axis=-1,
        max_axis=0,
    ),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="all"),
)
def test_all(
    *,
    dtype_x_axis,
    keepdims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="all",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keepdims,
    )


# any
@handle_cmd_line_args()
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=ivy_np.valid_int_dtypes,
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_axis=-1,
        max_axis=0,
    ),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="any"),
)
def test_any(
    *,
    dtype_x_axis,
    keepdims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="any",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keepdims,
    )
