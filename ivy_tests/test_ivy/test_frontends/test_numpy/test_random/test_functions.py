# global
from hypothesis import strategies as st

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
def test_numpy_random(input_dtypes, num_positional_args, size, native_array):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="random.random",
        test_values=False,
        size=size,
    )


# dirichlet
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=2),
        shape=st.tuples(
            st.integers(min_value=2, max_value=5),
        ),
        min_value=0,
        max_value=100,
        exclude_min=True,
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.dirichlet"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_dirichlet(
    dtype_and_x,
    size,
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
        fn_tree="random.dirichlet",
        alpha=x[0],
        test_values=False,
        size=size,
    )
