# global
import random

import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _dtype_x_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    if random.randint(0, 9) % 2 != 0:
        axis = None

    where = draw(
        helpers.array_values(
            dtype=ivy.bool,
            shape=shape,
        )
    )

    where = draw(st.sampled_from([where, np._NoValue]))
    return (dtype, x, axis), where


# sum
@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_axis(
        available_dtypes=ivy_np.valid_float_dtypes, min_num_dims=2, min_dim_size=2
    ),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    keep_dims=st.booleans(),
    initial=st.sampled_from([np._NoValue, 0]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.sum"
    ),
)
def test_numpy_sum(
    dtype_x_axis,
    dtype,
    keep_dims,
    initial,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    (input_dtype, x, axis), where = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="sum",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        dtype=dtype,
        keepdims=keep_dims,
        out=None,
        initial=initial,
        where=where,
    )
