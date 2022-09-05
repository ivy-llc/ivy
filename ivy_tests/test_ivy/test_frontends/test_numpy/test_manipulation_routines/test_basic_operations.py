# global
import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# shape
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.shape"
    ),
    native_array=helpers.array_bools(num_arrays=1),
)
def test_numpy_shape(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = xs_n_input_dtypes_n_unique_idx
    xs = np.asarray(xs, dtype=input_dtypes)
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        with_out=False,
        fw=fw,
        frontend="numpy",
        fn_tree="shape",
        array=xs,
    )
