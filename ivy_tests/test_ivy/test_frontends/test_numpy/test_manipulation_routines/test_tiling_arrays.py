# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# tile
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    dtype_and_repeats=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape").map(
            lambda rep: (len(rep),)
        ),
        min_value=0,
        max_value=10,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.tile"
    ),
)
def test_numpy_tile(
    dtype_and_x,
    dtype_and_repeats,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    repeats_dtype, repeats = dtype_and_repeats
    helpers.test_frontend_function(
        input_dtypes=[input_dtype, repeats_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="tile",
        A=np.array(x, dtype=input_dtype),
        reps=np.array(repeats, dtype=repeats_dtype),
    )


# repeat
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    repeats=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.repeat"
    ),
)
def test_numpy_repeat(
    dtype_and_x,
    repeats,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="repeat",
        a=np.array(x, dtype=input_dtype),
        repeats=repeats,
        axis=axis,
    )
