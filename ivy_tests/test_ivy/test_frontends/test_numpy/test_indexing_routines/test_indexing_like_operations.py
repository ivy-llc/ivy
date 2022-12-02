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
import ivy


@st.composite
def st_dtype_arr_and_axes(draw):
    dtypes, xs, x_shape = draw(
        helpers.dtype_and_values(
            num_arrays=1,
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=st.shared(
                helpers.get_shape(
                    allow_none=False,
                    min_num_dims=2,
                    max_num_dims=5,
                    min_dim_size=2,
                    max_dim_size=10,
                )
            ),
            ret_shape=True,
        )
    )
    axis1, axis2 = draw(
        helpers.get_axis(
            shape=x_shape,
            sorted=False,
            unique=True,
            min_size=2,
            max_size=2,
            force_tuple=True,
        )
    )
    return dtypes[0], xs[0], axis1, axis2


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
    # dtype_arr_and_axes=st_dtype_arr_and_axes(),
    dtype_and_axis=helpers.get_axis(
        shape=st.shared(
            helpers.get_shape(
                allow_none=False,
                min_num_dims=2,
                max_num_dims=5,
                min_dim_size=2,
                max_dim_size=10,
            )
        ),
        unique=True,
        max_size=2,
        min_size=2,
        force_tuple=True,
    ),
    offset=st.integers(min_value=-1, max_value=1),
)
def test_numpy_diagonal(
    # dtype_arr_and_axes,
    dtype_and_x,
    dtype_and_axis,
    as_variable,
    num_positional_args,
    native_array,
    # where,
    with_out,
    offset,
    # axis1,
    # dtype_x_axis,
    # axis2,
    # dtype_and_axis2,
    on_device,
    fn_tree,
    frontend,
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    axis = dtype_and_axis
    print("axis", axis)

    # axes = dtype_and_axis
    # print(axes)
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
        axis1=axis[0],
        axis2=axis[1],
    )
