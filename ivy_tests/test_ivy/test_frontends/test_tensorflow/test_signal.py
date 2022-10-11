import ivy
from hypothesis import given, strategies as st
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# hann_window


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.hann_window",
    ),
)
# helpers.test_frontend_function to test ivy.hann_window
def test_hann_window(dtype_and_x, num_positional_args):
    dtype, x = dtype_and_x
    ivy.hann_window(x, dtype=dtype)
