from hypothesis import given
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# kaiser_window
@handle_cmd_line_args
@given(
    dtype_and_window_length=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")
    ),
    dtype_and_beta=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.signal.kaiser_window"
    ),
    dtype=helpers.get_dtypes("numeric"),
)
def test_tensorflow_kaiser_window(
    dtype_and_window_length,
    dtype_and_beta,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
):
    window_length_dtype, window_length = dtype_and_window_length
    beta_dtype, beta = dtype_and_beta
    helpers.test_frontend_function(
        input_dtypes=[window_length_dtype[0], beta_dtype[0]],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="signal.kaiser_window",
        window_length=window_length,
        beta=beta,
    )
