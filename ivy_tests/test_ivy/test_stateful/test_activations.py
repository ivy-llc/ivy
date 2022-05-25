"""Collection of tests for unified neural network activations."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
from ivy_tests.test_ivy import helpers

import ivy
import ivy.functional.backends.numpy as ivy_np

#GELU
@given(
    x=st.lists(st.floats()),
    dtype=st.sampled_from(ivy.float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    class_method=st.booleans(),
)
def test_gelu( x, dtype, as_variable, with_out,class_method, num_positional_args, native_array, container, fw,):
    if dtype in ivy.invalid_dtypes:
        return  # invalid dtype
    if dtype == "float16" and fw == "torch":
        return  # torch does not support float16 for gelu
    if dtype == "bfloat16":
        return  # bfloat16 is not supported by numpy
    #class_method = ivy.GELU
    helpers.test_array_class(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        False,
        fw,
        "GELU",
        #x=np.asarray(x, dtype=dtype),
    )


@given(
    dtype=st.sampled_from(ivy.float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    class_method=st.booleans(),
)
def test_geglu( dtype, as_variable, with_out,class_method, num_positional_args, native_array, container, fw,):
    helpers.test_array_class(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        class_method,
        fw,
        "GEGLU",
    )
