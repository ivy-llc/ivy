# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.torch as ivy_torch


# one_hot
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_torch.valid_numeric_dtypes),
    depth=st.integers(min_value=3, max_value=5),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="one_hot"),
    native_array=st.booleans(),
)
def test_torch_one_hot(
    dtype_and_x,
    depth,
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
        fn_tree="one_hot",
        test_values=False,
        indices=np.asarray([1, 2, 0], dtype=input_dtype),
        depth=depth,
    )

