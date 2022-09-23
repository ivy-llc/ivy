# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# transpose
@handle_cmd_line_args
@given(
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=2, max_num_dims=5,
        min_dim_size=1, max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.transpose"
    ),
)
def test_numpy_transpose(
    array_and_axes,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    array, axes = array_and_axes
    helpers.test_frontend_function(
        input_dtypes=["int8"],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="transpose",
        array=np.array(array),
        axes=axes,
    )

