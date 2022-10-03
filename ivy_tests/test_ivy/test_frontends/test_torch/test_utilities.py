# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# can_cast
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True), num_arrays=1
    ),
    to_dtype=helpers.get_dtypes("valid", full=False),
    num_positional_args=helpers.num_positional_args(fn_name="can_cast"),
)
def test_can_cast(
    dtype_and_x,
    to_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="can_cast",
        from_=x[0],
        to=to_dtype[0],
    )
