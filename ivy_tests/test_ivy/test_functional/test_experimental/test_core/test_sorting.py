# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# msort
@handle_test(
    fn_tree="functional.ivy.experimental.msort",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        min_value=-100,
        max_value=100,
    ),
    test_gradients=st.just(False),
)
def test_msort(
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# lexsort
@handle_test(
    fn_tree="functional.ivy.experimental.lexsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_lexsort(
    dtype_x_axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        ground_truth_backend="numpy",
        test_flags=test_flags,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        keys=x[0],
        axis=axis,
    )
