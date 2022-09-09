# global
# import ivy
import numpy as np

from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers

# import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.tril"
    ),
)
def test_numpy_tril(
    dtype_and_x,
    k,
    num_positional_args,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="numpy",
        fn_tree="tril",
        m=np.asarray(x),
        k=k,
    )
