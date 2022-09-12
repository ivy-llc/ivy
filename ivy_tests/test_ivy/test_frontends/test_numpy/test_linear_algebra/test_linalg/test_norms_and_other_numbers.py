# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
)


@handle_cmd_line_args
@given(
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=5,
        min_axis=-2,
        max_axis=1,
    ),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.norm"
    ),
)
def test_numpy_norm(
    dtype_values_axis, keepdims, as_variable, native_array, num_positional_args, fw
):
    dtype, x, axis = dtype_values_axis
    if len(np.shape(x)) == 1:
        axis = None
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.norm",
        x=np.array(x, dtype=dtype),
        ord=None,
        axis=axis,
        keepdims=keepdims,
    )


# matrix_rank


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        min_value=-1e05,
        max_value=1e05,
    ),
    rtol=st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.matrix_rank"
    ),
)
def test_numpy_matrix_rank(
    dtype_and_x, rtol, as_variable, native_array, num_positional_args, fw
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
        fn_tree="linalg.matrix_rank",
        A=np.array(x, dtype=dtype),
        tol=rtol,
    )


# det


@handle_cmd_line_args
@given(
    dtype_and_x=_get_dtype_and_matrix(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.det"
    ),
)
def test_numpy_det(dtype_and_x, as_variable, native_array, num_positional_args, fw):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.det",
        a=np.array(x, dtype=dtype),
    )


# slogdet


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=20).map(lambda x: tuple([x, x])),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.slogdet"
    ),
)
def test_numpy_slogdet(dtype_and_x, as_variable, native_array, num_positional_args, fw):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.slogdet",
        a=np.array(x, dtype=dtype),
    )
