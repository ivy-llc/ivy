# local
import ivy_tests.test_ivy.helpers as helpers
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# squeeze
@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="value_shape"))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0, None)

    return draw(st.sampled_from(valid_axes))


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.squeeze"
    ),
)
def test_numpy_squeeze(
    dtype_and_x,
    axis,
    num_positional_args,
    fw,
):
    input_dtype, x = dtype_and_x
    input_dtype = [input_dtype]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        native_array_flags=False,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="squeeze",
        a=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.expand_dims"
    ),
)
def test_numpy_expand_dims(
    dtype_and_x,
    axis,
    num_positional_args,
    fw,
):
    input_dtype, x = dtype_and_x
    input_dtype = [input_dtype]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        native_array_flags=False,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="numpy",
        fn_tree="expand_dims",
        a=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
    )
