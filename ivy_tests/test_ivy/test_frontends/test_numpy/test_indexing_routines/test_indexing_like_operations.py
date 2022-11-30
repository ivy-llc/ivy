# Testing Function
# global
from hypothesis import strategies as st

import ivy

# local
import ivy_tests.test_ivy.helpers as helpers


# from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
#   statistical_dtype_values,
# )
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

# import ivy


@handle_frontend_test(
    fn_tree="numpy.diagonal",
    # dtype_x_axis=helpers.dtype_values_axis(
    #   available_dtypes=helpers.get_dtypes("numeric"),
    #  num_arrays=1,
    # shared_dtype = True,
    # min_num_dims = 2
    # ),
    # dtype_and_x=helpers.dtype_and_values(
    #   available_dtypes=helpers.get_dtypes("numeric"),
    #  num_arrays=1,
    # min_value=0,
    # max_value = 1,
    # shared_dtype=True,
    # min_num_dims=2,
    # ),
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        shared_dtype=True,
        min_num_dims=2,
        min_axis=-2,
        max_axis=1,
    ),
    # dtype_and_axis=helpers.get_axis(
    # available_dtypes=helpers.get_dtypes("numeric"),
    #   shape=(2, 3),
    #  min_size=2,
    # max_size = 2,
    # ),
    # dtype_and_axis2=helpers.get_axis(
    # available_dtypes=helpers.get_dtypes("numeric"),
    #   shape=(2, 3),
    #   min_size=1,
    # ),
    # where=np_frontend_helpers.where(),
    # dtype_and_axis1=helpers.dtype_values_axis(
    #   available_dtypes=helpers.get_dtypes("numeric"),
    # num_arrays=1,
    # min_value=0,
    # max_value = 1,
    #  shared_dtype=True,
    # min_num_dims=2,
    # ),
    offset=st.integers(min_value=-1, max_value=1),
    # axis1=st.integers(min_value=-2, max_value=2),
    # axis2=st.integers(min_value=-2, max_value=2),
    # axis1!=axis2,
    # dtype=helpers.get_dtypes("float", full=False, none=True),
    # where=np_frontend_helpers.where(),
)
def test_numpy_diagonal(
    # dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    # where,
    with_out,
    offset,
    # axis1,
    dtype_x_axis,
    # axis2,
    # dtype_and_axis2,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    # axis = dtype_and_axis
    # print ('axis',axis)
    # axis2 = dtype_and_axis[1]
    new_axis = 0
    if axis < 0:
        new_axis = axis + 1
    else:
        new_axis = axis - 1
    as_variable = as_variable

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        # where=where,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        # fw=fw,
        # frontend="numpy",
        # fn_tree="diagonal",
        a=ivy.native_array(x, dtype=ivy.int32),
        offset=offset,
        axis1=new_axis,
        axis2=axis,
    )
