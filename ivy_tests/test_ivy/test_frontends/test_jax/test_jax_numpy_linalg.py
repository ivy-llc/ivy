# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# svd
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=2, min_dim_size=2
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.svd"
    ),
    full_matrices=st.booleans(),
)
def test_jax_svd(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    full_matrices,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        test_values=False,
        frontend="jax",
        fn_tree="numpy.linalg.svd",
        x=np.asarray(x, dtype=input_dtype),
        full_matrices=full_matrices,
    )
