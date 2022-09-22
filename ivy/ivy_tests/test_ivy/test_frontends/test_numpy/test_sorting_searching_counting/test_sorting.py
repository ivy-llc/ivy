# global
import random

import numpy as np
from hypothesis import given, strategies as st

# local
# import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _dtype_x_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    if random.randint(0, 9) % 2 != 0:
        axis = None

    return dtype, x, axis


# sum
@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_axis(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1, min_dim_size=2
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.argsort"
    ),
)
def test_numpy_argsort(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        test_values=False,
        frontend="numpy",
        fn_tree="argsort",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
    )
