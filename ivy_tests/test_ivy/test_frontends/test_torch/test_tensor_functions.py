# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# is_nonzero
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_dtypes).intersection(
                set(ivy_torch.valid_dtypes)
            )
        ),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.is_nonzero"
    ),
)
def test_torch_is_nonzero(
    dtype_and_x,
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
        frontend="torch",
        fn_tree="is_nonzero",
        input=np.asarray(x, dtype=input_dtype),
    )