"""Collection of tests for unified neural network activation functions."""

# global
import numpy as np
from hypothesis import given, strategies as st
# local

import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np



# relu
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=st.integers(0, 2),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_relu(dtype_and_x, as_variable, with_out, native_array,num_positional_args,container,instance_method,fw):
        dtype, x = dtype_and_x
        x = np.asarray(x, dtype=dtype)
        if x.shape == ():
            return
        if fw == "torch" and dtype == "float16":
            return
        helpers.test_array_function(
            dtype,
            as_variable,
            with_out,
            native_array,
            fw,
            num_positional_args,
            container,
            instance_method,
            "relu",
            x=x,
        )


# leaky_relu
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    native_array=st.booleans(),
    num_positional_args=st.integers(0, 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
    alpha = st.floats(),
)

def test_leaky_relu(dtype_and_x, alpha,as_variable, num_positional_args,container,instance_method,native_array, fw):
    dtype,x = dtype_and_x
    if not ivy.all(ivy.isfinite(ivy.array(x))) or not ivy.isfinite(ivy.array([alpha])):
        return
    if fw == "torch" and dtype == "float16" :
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "leaky_relu",
        x=np.asarray(x, dtype=dtype),
        alpha= alpha,
    )


# gelu
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=st.integers(0,2),
    container=st.booleans(),
    instance_method=st.booleans(),
    approximate =st.booleans(),
)
def test_gelu(dtype_and_x,as_variable,approximate, num_positional_args,container,instance_method,native_array, fw):
    dtype,x = dtype_and_x
    if fw == "torch" and dtype == "float16":
        return
    x = np.asarray(x, dtype=dtype)
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "gelu",
        x=x,
        approximate = approximate,
    )

# tanh
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=st.integers(0, 2),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_tanh(dtype_and_x,as_variable, num_positional_args,container,instance_method,native_array, fw):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "tanh",
        x=np.asarray(x, dtype=dtype),
    )


# sigmoid
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=st.integers(0, 2),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_sigmoid(dtype_and_x,as_variable, num_positional_args,container,instance_method,native_array, fw):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "sigmoid",
        x=np.asarray(x, dtype=dtype),
    )
# softmax
@given(
    dtype_and_x=helpers.dtype_and_values(ivy.all_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=st.integers(0, 2),
    container=st.booleans(),
    instance_method=st.booleans(),


)
def test_softmax(dtype_and_x,as_variable, num_positional_args,container,instance_method,native_array, fw):
    dtype, x = dtype_and_x
    axis = -1
    if fw == "torch" and dtype == "float16":
        return
    x = np.asarray(x, dtype=dtype)
    if x.shape == ():
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "softmax",
        x=x,
        axis = axis,
    )
# softplus
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=st.integers(0, 2),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_softplus(dtype_and_x,as_variable, num_positional_args,container,instance_method,native_array, fw):
    dtype, x = dtype_and_x
    if fw == "torch" and dtype == "float16":
        return
    helpers.test_array_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "softplus",
        x=np.asarray(x, dtype=dtype),
)
