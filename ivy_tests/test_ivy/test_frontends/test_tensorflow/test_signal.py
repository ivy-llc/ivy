import ivy
from hypothesis import given, strategies as st
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# hann_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    target_dtype=helpers.get_dtypes("float"),
    periodic=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.signal.hann_window"
    ),
)
def test_tensorflow_hann_window(
    dtype_and_x, target_dtype, as_variable, 
    periodic, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="signal.hann_window",
        window_length=x[0],
        periodic=periodic,
        dtype=target_dtype,
    )

