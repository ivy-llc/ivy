# global

import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
)


# outer
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.outer"
    ),
)
def test_numpy_outer(
    dtype_and_x,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="outer",
        a=xs[0],
        b=xs[1],
    )


# inner
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.inner"
    ),
)
def test_numpy_inner(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="inner",
        a=xs[0],
        b=xs[1],
    )


# matmul
@handle_cmd_line_args
@given(
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.matmul"
    ),
)
def test_numpy_matmul(
    x, y, as_variable, with_out, native_array, num_positional_args, fw
):
    dtype1, x1 = x
    dtype2, x2 = y
    helpers.test_frontend_function(
        input_dtypes=[dtype1, dtype2],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="matmul",
        x1=np.array(x1, dtype=dtype1),
        x2=np.array(x2, dtype=dtype2),
    )


# matrix_power
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=8).map(lambda x: tuple([x, x])),
    ),
    n=helpers.ints(min_value=1, max_value=8),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.matrix_power"
    ),
)
def test_numpy_matrix_power(
    dtype_and_x, n, as_variable, native_array, num_positional_args, fw
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.matrix_power",
        a=x[0],
        n=n,
    )
