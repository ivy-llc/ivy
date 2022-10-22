# global
from hypothesis import given, strategies as st

# local
import numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# isin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric"),
                                         num_arrays=2),
    num_positional_args=helpers.num_positional_args(fn_name="isin"),
    invert=st.booleans(),
    assume_unique=st.booleans()
)
def test_isin(
    dtype_and_x,
    with_out,
    invert,
    assume_unique,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, values = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        instance_method=instance_method,
        container_flags=container,
        fw=fw,
        fn_name="isin",
        element=np.asarray(values[0], dtype=dtype[0]),
        test_elements=np.asarray(values[1], dtype=dtype[1]),
        invert=invert,
        assume_unique=assume_unique,
    )
