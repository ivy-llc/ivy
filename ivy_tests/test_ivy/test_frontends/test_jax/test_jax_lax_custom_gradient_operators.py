import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.stop_gradient"
    ),
)
def test_stop_gradient(dtype_and_x, as_variable, num_positional_args, native_array, fw):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        with_out=False,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.stop_gradient",
        x=np.asarray(x, dtype=dtype),
    )
