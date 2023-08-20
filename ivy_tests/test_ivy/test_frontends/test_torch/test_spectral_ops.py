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
    dtype_and_coefficients=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_num_dims=0,
        num_arrays=2,
        min_value=0,
        max_value=5,
    ),
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("float"),
    test_with_out=st.just(False),
)
def test_torch_hamming_window(
    dtype_and_window_length,
    dtype_and_coefficients,
    periodic,
    *,
    dtype,
    fn_tree,
    frontend,
    test_flags,
    backend_fw
):
    window_length_dtype, window_length = dtype_and_window_length
    coefficients_dtypes, coefficients = dtype_and_coefficients

    helpers.test_frontend_function(
        input_dtypes=window_length_dtype + coefficients_dtypes,
        window_length=int(window_length[0]),
        alpha=float(coefficients[0]),
        beta=float(coefficients[1]),
        periodic=periodic,
        dtype=dtype[0],
        fn_tree=fn_tree,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        rtol=1e-2,
        atol=1e-2,
    )
