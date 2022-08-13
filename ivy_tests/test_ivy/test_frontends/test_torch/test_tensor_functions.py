import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch

#is_tensor
@given(
dtype_and_x=helpers.dtype_and_values(
    available_dtypes=tuple(
        set(ivy_np.valid_float_dtypes).intersection(
            set(ivy_torch.valid_float_dtypes)
        )
    )
),
as_variable=st.booleans(),
with_out=st.booleans(),
num_positional_args=helpers.num_positional_args(
    fn_name="functional.frontends.torch.is_tensor"
),
native_array=st.booleans(),
)
def test_torch_is_tensor(
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
        fn_name="is_tensor",
        input=np.asarray(x, dtype=input_dtype),
        out=None,
    )