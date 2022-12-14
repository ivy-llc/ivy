# global
import numpy as np
from hypothesis import assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# unique_values
@handle_test(
    fn_tree="functional.ivy.unique_values",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_unique_values(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# unique_all
@handle_test(
    fn_tree="functional.ivy.unique_all",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
)
def test_unique_all(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# unique_counts
@handle_test(
    fn_tree="functional.ivy.unique_counts",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
)
def test_unique_counts(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# unique_inverse
@handle_test(
    fn_tree="functional.ivy.unique_inverse",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
)
def test_unique_inverse(
    *,
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )
