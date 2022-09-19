# global
import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.isneginf"
    ),
)
def test_numpy_isneginf(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="isneginf",
        x=np.array(x, dtype=dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=np.inf
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.isposinf"
    ),
)
def test_numpy_isposinf(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="isposinf",
        x=np.array(x, dtype=dtype),
    )
