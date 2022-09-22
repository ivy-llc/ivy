"""Collection of tests for searching functions."""

# Global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Helpers #
############


@st.composite
def _dtype_x_limited_axis(draw, *, allow_none=False):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=1,
            min_dim_size=1,
            ret_shape=True,
        )
    )
    if allow_none and draw(st.booleans()):
        return dtype, x, None

    axis = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    return dtype, x, axis


@st.composite
def _broadcastable_trio(draw):
    shape = draw(helpers.get_shape(min_num_dims=1, min_dim_size=1))
    cond = draw(helpers.array_values(dtype="bool", shape=shape))
    dtypes, xs = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=2,
            shape=shape,
            large_abs_safety_factor=16,
            small_abs_safety_factor=16,
            safety_factor_scale="log",
        )
    )
    return cond, xs[0], xs[1], dtypes[0], dtypes[1]


# Functions #
#############


@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_limited_axis(allow_none=True),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="argmax"),
)
def test_argmax(
    *,
    dtype_x_axis,
    keepdims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="argmax",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_limited_axis(allow_none=True),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="argmin"),
)
def test_argmin(
    *,
    dtype_x_axis,
    keepdims,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="argmin",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer", full=True),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    as_tuple=st.booleans(),
    size=st.integers(min_value=1, max_value=5),
    fill_value=st.integers(min_value=0, max_value=5),
    num_positional_args=helpers.num_positional_args(fn_name="nonzero"),
)
def test_nonzero(
    *,
    dtype_and_x,
    as_tuple,
    size,
    fill_value,
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
        fn_name="nonzero",
        x=np.asarray(x, dtype=input_dtype),
        as_tuple=as_tuple,
        size=size,
        fill_value=fill_value,
    )


@handle_cmd_line_args
@given(
    broadcastables=_broadcastable_trio(),
    num_positional_args=helpers.num_positional_args(fn_name="where"),
)
def test_where(
    *,
    broadcastables,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    cond, x1, x2, dtype1, dtype2 = broadcastables

    helpers.test_function(
        input_dtypes=["bool", dtype1, dtype2],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="where",
        condition=np.asarray(cond, dtype="bool"),
        x1=np.asarray(x1, dtype=dtype1),
        x2=np.asarray(x2, dtype=dtype2),
    )


# argwhere
@handle_cmd_line_args
@given(
    x=helpers.dtype_and_values(available_dtypes=("bool",)),
    num_positional_args=helpers.num_positional_args(fn_name="argwhere"),
)
def test_argwhere(
    *,
    x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="argwhere",
        x=np.asarray(x, dtype=dtype),
    )
