"""Collection of tests for utility functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# all
@given(
    data=st.data(),
    keepdims=st.booleans(),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_all(
    data,
    keepdims,
    input_dtype,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    num_positional_args = data.draw(helpers.num_positional_args(fn_name="all"))
    shape = data.draw(helpers.get_shape(min_num_dims=1))
    x = data.draw(helpers.array_values(dtype=input_dtype, shape=shape))
    axis = data.draw(helpers.get_axis(shape=shape))
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "all",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keepdims,
    )


# any
@given(
    data=st.data(),
    keepdims=st.booleans(),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_any(
    data,
    keepdims,
    input_dtype,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    num_positional_args = data.draw(helpers.num_positional_args(fn_name="any"))
    shape = data.draw(helpers.get_shape(min_num_dims=1))
    x = data.draw(helpers.array_values(dtype=input_dtype, shape=shape))
    axis = data.draw(helpers.get_axis(shape=shape))
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "any",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keepdims,
    )
