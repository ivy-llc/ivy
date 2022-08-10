# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# flip
@given(
    dtype_values_axis = helpers.dtype_values_axis(
        available_dtypes=tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_torch.valid_float_dtypes))),
        shape=helpers.get_shape(min_num_dims=1)
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.flip"),
    native_array=st.booleans(),
)
def test_torch_flip(
    *,
    dtype_values_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_name="flip",
        input=np.asarray(value, dtype=input_dtype),
        dims=axis,
    )
