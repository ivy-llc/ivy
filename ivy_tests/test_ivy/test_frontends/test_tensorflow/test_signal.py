# global
import ivy
import numpy as np
from hypothesis import given, assume, strategies as st
from ivy.functional.backends.numpy.data_type import dtype

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# hann_window


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.hann_window",
    ),
)
def test_hann_window(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x , y = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="hann_window",
        window_length=x,
        periodic=True,
        dtype=y

    )
