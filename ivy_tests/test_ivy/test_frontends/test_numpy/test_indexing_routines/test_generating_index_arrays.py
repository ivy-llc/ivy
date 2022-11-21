from hypothesis import given, strategies as st

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


def test_numpy_indices(
    dtype_and_x,
    pred_cond,
    num_positional_args,
    as_variable,
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
        fn_tree="indices",
        pred=pred_cond,
        operand=x[0],
        sparse=False,
    )
