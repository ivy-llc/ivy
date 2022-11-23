# Testing Function
# global
from hypothesis import strategies as st

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
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        # min_value=0,
        # max_value = 1,
        shared_dtype=True,
        min_num_dims=2,
    ),
    where=np_frontend_helpers.where(),
    # dtype_and_axis1=helpers.dtype_values_axis(
    #   available_dtypes=helpers.get_dtypes("numeric"),
    # num_arrays=1,
    # min_value=0,
    # max_value = 1,
    #  shared_dtype=True,
    # min_num_dims=2,
    # ),
    offset=st.integers(min_value=0),  # ,max_value=10),
    axis1=st.integers(min_value=-2, max_value=2),
    axis2=st.integers(min_value=-2, max_value=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.diagonal"
    ),
    # dtype=helpers.get_dtypes("float", full=False, none=True),
    # where=np_frontend_helpers.where(),
)
def test_diagonal(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    where,
    offset,
    axis1,
    axis2,
    with_out,
):
    input_dtype, x = dtype_and_x
    as_variable = as_variable

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        where=where,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        # fw=fw,
        frontend="numpy",
        fn_tree="diagonal",
        x=x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )
