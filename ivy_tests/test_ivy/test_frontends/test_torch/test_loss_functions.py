# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# cross_entropy
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(fn_name="cross_entropy"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_torch_cross_entropy(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_name="cross_entropy",
        input=np.asarray(x[0], dtype=input_dtype[0]),
        target=np.asarray(x[1], dtype=input_dtype[1]),
    )
