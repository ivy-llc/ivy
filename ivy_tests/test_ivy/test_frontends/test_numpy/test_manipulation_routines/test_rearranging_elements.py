import ivy.functional.backends.numpy as ivy_np

# local
from hypothesis import given
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import numpy as np
import ivy_tests.test_ivy.helpers as helpers


# roll
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=2,
        min_dim_size=2,
    ),
    shift=helpers.ints(min_value=1, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.roll"
    ),
)
def test_numpy_roll(
    dtype_and_x,
    shift,
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
        fn_tree="roll",
        a=np.array(x, dtype=input_dtype),
        shift=shift,
        axis=axis,
    )
