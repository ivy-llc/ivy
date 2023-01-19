# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# kaiser_window
@handle_frontend_test(
    fn_tree="tensorflow.signal.kaiser_window",
    dtype_and_window_length=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")
    ),
    dtype_and_beta=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric"),
    test_with_out=st.just(False),
)
def test_tensorflow_kaiser_window(
    *,
    dtype_and_window_length,
    dtype_and_beta,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    window_length_dtype, window_length = dtype_and_window_length
    beta_dtype, beta = dtype_and_beta
    helpers.test_frontend_function(
        input_dtypes=[window_length_dtype[0], beta_dtype[0]],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        window_length=window_length,
        beta=beta,
        dtype=dtype,
    )
