# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args, given


# random
@handle_cmd_line_args
@given(
    input_dtypes=helpers.get_dtypes("integer", full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.random"
    ),
    size=helpers.get_shape(allow_none=True),
)
def test_numpy_random(input_dtypes, num_positional_args, size, fw, native_array):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        test_values=False,
        fw=fw,
        frontend="numpy",
        fn_tree="random.random",
        size=size,
    )


# dirichlet
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=1,
        max_num_dims=1,

    ),
    as_variable=helpers.array_bools(),
    size=st.one_of(st.tuples(), st.integers()),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.random"
    ),
    native_array=helpers.array_bools(),)
def test_numpy_dirichlet(
    dtype_and_x,
    size,
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
        frontend="numpy",
        fn_tree="random.dirichlet",
        alpha=np.asarray(x, dtype=input_dtype[0]),
        size=size,
    )
