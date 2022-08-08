# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# flip
@given(
    dtype_value=
        helpers.dtype_and_values(
            available_dtypes=tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_torch.valid_float_dtypes))),
            shape=st.shared(
                helpers.get_shape(
                    min_num_dims=1
                ),
                key='shape'
            ),
        ),
    dims=
        helpers.get_axis(
            shape=st.shared(
                helpers.get_shape(
                    min_num_dims=1
                ),
                key='shape'
            ),
            min_size=1,
            max_size=1,
        ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.flip"),
    native_array=st.booleans(),
    data=st.data(),
)
def test_torch_flip(
    *,
    data,
    dtype_value,
    dims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_name="flip",
        input=np.asarray(value, dtype=input_dtype),
        dims=dims,
    )
