# global
import numpy as np
import ivy
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


# roll
@given(
    dtype_values = helpers.dtype_and_values(
        available_dtypes=tuple(
                set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_torch.valid_float_dtypes))
        ),
        shape=st.shared(
            helpers.get_shape(min_num_dims=1),
            key='value_shape',
        ),
    ),
    shift=helpers.dtype_and_values(
        available_dtypes=[ivy.int32, ivy.int64],
        max_num_dims=1,
        min_dim_size=st.shared(st.integers(1, 2147483647), key='shift_length'),
        max_dim_size=st.shared(st.integers(1, 2147483647), key='shift_length')
    ),
    axis=helpers.get_axis(
        shape=st.shared(
            helpers.get_shape(min_num_dims=1),
            key='value_shape',
        ),
        unique=False,
        min_size=st.shared(st.integers(1, 2147483647), key='shift_length'),
        max_size=st.shared(st.integers(1, 2147483647), key='shift_length')
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.roll"
    ),
    native_array=st.booleans(),
)
def test_torch_roll(
    dtype_values,
    shifts,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="roll",
        input=np.asarray(value, dtype=input_dtype),
        shifts=shifts,
        dims=axis,
    )
