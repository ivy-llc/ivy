# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# abs
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.abs"
    ),
)
def test_jax_numpy_abs(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.abs",
        x=x[0],
    )


# allclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.allclose"
    ),
    equal_nan=st.booleans(),
)
def test_jax_numpy_allclose(
    dtype_and_input,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.allclose",
        a=input[0],
        b=input[1],
        rtol=1e-05,
        atol=1e-08,
        equal_nan=equal_nan,
    )
