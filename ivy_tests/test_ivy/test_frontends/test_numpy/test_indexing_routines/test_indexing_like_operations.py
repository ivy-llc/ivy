# Testing Function
import ivy
import numpy as np
from hypothesis import given, strategies as st
from hypothesis import assume
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        #min_value=0,
        #max_value = 1,
        shared_dtype=True,
        min_num_dims=2,
    ),

    #dtype=helpers.get_dtypes("float", full=False, none=True),
    #where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.diagonal"
    ),
    offset = st.integers(min_value=0),#,max_value=10),

    axis1 = st.integers(min_value=0,max_value=1),
    axis2 = st.integers(min_value=0,max_value=1),
)
def test_numpy_diagonal(
        dtype_and_x,as_variable,num_positional_args,native_array,
        fw,offset,axis1,axis2):
    input_dtype, x = dtype_and_x
    as_variable = as_variable,

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="diagonal",
        a=x,
        offset=offset,
        axis1=axis1,
        axis2=axis2
    )