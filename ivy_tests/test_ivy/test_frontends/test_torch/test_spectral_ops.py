from hypothesis import strategies as st
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="torch.bartlett_window",
    window_length=helpers.ints(min_value=2, max_value=100),
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_bartlett_window(
    window_length,
    periodic,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        window_length=window_length,
        periodic=periodic,
        dtype=dtype[0],
        rtol=1e-02,
        atol=1e-02,
    )


@handle_frontend_test(
    window_length=helpers.ints(min_value=1, max_value=100),
    dtype=helpers.get_dtypes("float", full=False),
    fn_tree="torch.blackman_window",
    periodic=st.booleans(),
)
def test_torch_blackman_window(
    *,
    window_length,
    dtype,
    periodic,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        on_device=on_device,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        window_length=window_length,
        periodic=periodic,
        dtype=dtype[0],
        rtol=1e-02,
        atol=1e-02,
    )


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
    test_with_out=st.just(False),
)
def test_torch_hamming_window(
    dtype_and_window_length,
    periodic,
    dtype_and_coefficients,
    *,
    dtype,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    window_length_dtype, window_length = dtype_and_window_length
    coefficients_dtypes, coefficients = dtype_and_coefficients

    helpers.test_frontend_function(
        input_dtypes=window_length_dtype + coefficients_dtypes,
        window_length=int(window_length[0]),
        periodic=periodic,
        alpha=float(coefficients[0]),
        beta=float(coefficients[1]),
        dtype=dtype[0],
        fn_tree=fn_tree,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        rtol=1e-1,
        atol=1e-1,
    )


@handle_frontend_test(
    window_length=helpers.ints(min_value=1, max_value=100),
    dtype=helpers.get_dtypes("float", full=False),
    fn_tree="torch.kaiser_window",
    periodic=st.booleans(),
    beta=helpers.floats(min_value=1, max_value=20),
)
def test_torch_kaiser_window(
    *,
    window_length,
    dtype,
    periodic,
    beta,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        on_device=on_device,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        window_length=window_length,
        periodic=periodic,
        beta=beta,
        dtype=dtype[0],
        rtol=1e-02,
        atol=1e-02,
    )
