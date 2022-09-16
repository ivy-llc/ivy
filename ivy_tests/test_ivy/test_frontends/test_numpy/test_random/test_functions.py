# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args, given


#random
#  size=helpers.get_shape(allow_none=True),
@handle_cmd_line_args
@given(
    input_dtypes=helpers.get_dtypes("integer"),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.absolute"
    ),
    native_array=st.booleans(),
    with_out=st.booleans(),
)

def test_numpy_random(
    input_dtypes,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="random.random",
        size=size
    )
