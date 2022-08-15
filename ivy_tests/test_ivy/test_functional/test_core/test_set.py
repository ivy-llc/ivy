# global
import numpy as np
from hypothesis import strategies as st, given, assume, settings

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# unique_values
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="unique_values"),
    data=st.data(),
)
@settings(deadline=None)
@handle_cmd_line_args
def test_unique_values(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="unique_values",
        x=np.asarray(x, dtype=dtype),
    )


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="unique_all"),
    data=st.data(),
)
@handle_cmd_line_args
def test_unique_all(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="unique_all",
        x=np.asarray(x, dtype=dtype),
    )


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="unique_counts"),
    data=st.data(),
)
@handle_cmd_line_args
def test_unique_counts(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="unique_counts",
        x=np.asarray(x, dtype=dtype),
    )


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="unique_inverse"),
    data=st.data(),
)
@handle_cmd_line_args
def test_unique_inverse(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="unique_inverse",
        x=np.asarray(x, dtype=dtype),
    )
