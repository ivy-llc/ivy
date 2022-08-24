# global
# import ivy
# import numpy as np
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
