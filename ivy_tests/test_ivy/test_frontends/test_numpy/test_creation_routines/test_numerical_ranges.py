# global
# import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers

import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# arange
@handle_cmd_line_args
@given(
    start=helpers.ints(min_value=-50, max_value=0),
    stop=helpers.ints(min_value=1, max_value=50),
    step=helpers.ints(min_value=1, max_value=5),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.arange"
    ),
)
def test_numpy_arange(
    start,
    stop,
    step,
    dtype,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="numpy",
        fn_tree="arange",
        start=start,
        stop=stop,
        step=step,
        dtype=dtype,
    )


# linspace
@handle_cmd_line_args
@given(
    dtype_and_start=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=2,
        min_dim_size=2,
        min_value=-50,
        max_value=0,
    ),
    dtype_and_stop=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=2,
        min_dim_size=2,
        min_value=1,
        max_value=50,
    ),
    num=helpers.ints(min_value=5, max_value=50),
    endpoint=st.booleans(),
    retstep=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linspace"
    ),
)
def test_numpy_linspace(
    dtype_and_start,
    dtype_and_stop,
    num,
    endpoint,
    retstep,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    start_dtype, start = dtype_and_start
    stop_dtype, stop = dtype_and_stop
    helpers.test_frontend_function(
        input_dtypes=[start_dtype, stop_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="linspace",
        start=np.asarray(start, dtype=start_dtype),
        stop=np.asarray(stop, dtype=stop_dtype),
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=start_dtype,
    )
