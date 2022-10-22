# global
from builtins import tuple
import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix,
    _get_second_matrix,
)

# solve


@handle_cmd_line_args
@given(
    x=_get_first_matrix(),
    y=_get_second_matrix(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.solve"
    ),
)
def test_numpy_solve(x, y, as_variable, native_array, num_positional_args):
    dtype1, x1 = x
    dtype2, x2 = y
    helpers.test_frontend_function(
        input_dtypes=[dtype1, dtype2],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="linalg.solve",
        a=x1,
        b=x2,
    )


# inv


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_dim_size=6,
        max_dim_size=6,
        min_num_dims=2,
        max_num_dims=2,
    ).filter(lambda x: np.linalg.det(x[1][0]) != 0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.inv"
    ),
)
def test_numpy_inv(dtype_and_x, as_variable, native_array, num_positional_args):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="linalg.inv",
        a=x[0],
    )


# pinv


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.pinv"
    ),
)
def test_numpy_pinv(dtype_and_x, as_variable, native_array, num_positional_args):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="linalg.pinv",
        a=x[0],
    )


# tensorsolve
@handle_cmd_line_args
@given(
    dtype_and_vals=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        min_value=-1e05,
        max_value=1e05,
    ).filter(lambda x: (x[1][0].shape[-2] == x[1][0].shape[-1])
             and (np.prod(x[1][0].shape[x[1][1].ndim:]) == np.prod(x[1][0].shape[:x[1][1].ndim]))),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.tensorsolve"
    ),
)
def test_numpy_tensorsolve(
    dtype_and_vals, as_variable, native_array, num_positional_args
):
    dtypes, x = dtype_and_vals
    x1, x2 = x
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="linalg.tensorsolve",
        a=x1,
        b=x2,
    )
