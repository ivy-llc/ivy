# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# hamming_window
@handle_frontend_test(
    fn_tree="torch.hamming_window",
    dtype_and_window_length=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=1,
        max_value=20,
    ),
    periodic=st.booleans(),
    dtype_and_coefficients=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_num_dims=0,
        num_arrays=2,
        min_value=0,
        max_value=5,
    ),
    dtype=helpers.get_dtypes("float"),
    requires_grad=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_hamming_window(
    dtype_and_window_length,
    periodic,
    dtype_and_coefficients,
    *,
    dtype,
    requires_grad,
    fn_tree,
    frontend,
    test_flags,
):
    window_length_dtype, window_length = dtype_and_window_length
    coefficients_dtypes, coefficients = dtype_and_coefficients
    helpers.test_frontend_function(
        input_dtypes=window_length_dtype + coefficients_dtypes,
        test_flags=test_flags,
        window_length=int(window_length[0]),
        periodic=periodic,
        alpha=float(coefficients[0]),
        beta=float(coefficients[1]),
        dtype=dtype[0],
        requires_grad=requires_grad,
        fn_tree=fn_tree,
        frontend=frontend,
    )
