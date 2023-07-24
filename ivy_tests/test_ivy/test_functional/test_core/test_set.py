# global
import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# unique_values
@handle_test(
    fn_tree="functional.ivy.unique_values",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    test_gradients=st.just(False),
)
def test_unique_values(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# unique_all
@handle_test(
    fn_tree="functional.ivy.unique_all",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    none_axis=st.booleans(),
    by_value=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    ground_truth_backend="numpy",
)
def test_unique_all(
    *, dtype_x_axis, none_axis, by_value, test_flags, backend_fw, fn_name, on_device
):
    dtype, x, axis = dtype_x_axis
    if none_axis:
        axis = None
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
        axis=axis,
        by_value=by_value,
    )


# unique_counts
@handle_test(
    fn_tree="functional.ivy.unique_counts",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_unique_counts(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# unique_inverse
@handle_test(
    fn_tree="functional.ivy.unique_inverse",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_unique_inverse(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x, 0.0)))

    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )
