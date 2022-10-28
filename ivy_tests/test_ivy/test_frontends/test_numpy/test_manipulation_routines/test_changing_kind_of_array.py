# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# asarray
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.asarray"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_asarray(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="asarray",
        a=x,
        dtype=input_dtype,
    )
