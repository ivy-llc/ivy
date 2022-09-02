# global
import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# matrix_power


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=8).map(lambda x: tuple([x, x])),
    ),
    n=helpers.ints(min_value=1, max_value=8),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.matrix_power"
    )
)
def test_numpy_matrix_power(dtype_and_x, n, as_variable, native_array, num_positional_args, fw):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="linalg.matrix_power",
        a=np.array(x, dtype=dtype),
        n=n
    )
