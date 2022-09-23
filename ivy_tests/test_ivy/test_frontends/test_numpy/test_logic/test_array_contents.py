# global
import numpy as np
from hypothesis import given, strategies as st

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


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True
    ),
    equal_nan=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.allclose"
    )
)
def test_numpy_allclose(
    dtype_and_x,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    fw,
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
        fn_tree="allclose",
        a=np.asarray(x[0], dtype=dtype[0]),
        b=np.asarray(x[1], dtype=dtype[1]),
        rtol=1e-05,
        atol=1e-08,
        equal_nan=equal_nan,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True
    ),
    equal_nan=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.isclose"
    )
)
def test_numpy_isclose(
    dtype_and_x,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    fw,
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
        fn_tree="isclose",
        a=np.asarray(x[0], dtype=dtype[0]),
        b=np.asarray(x[1], dtype=dtype[1]),
        rtol=1e-05,
        atol=1e-08,
        equal_nan=equal_nan,
    )
