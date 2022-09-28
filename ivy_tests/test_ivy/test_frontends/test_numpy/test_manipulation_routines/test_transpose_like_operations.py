# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# transpose
@handle_cmd_line_args
@given(
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.transpose"
    ),
)
def test_numpy_transpose(
    array_and_axes,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="transpose",
        array=np.array(array, dtype=dtype),
        axes=axes,
    )


# swapaxes
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
            unique=False,
            min_size=2,
            max_size=2,
            force_tuple=True,
        )
    )
    return dtypes[0], xs[0], axis1, axis2


@handle_cmd_line_args
@given(
    dtype_arr_and_axes=st_dtype_arr_and_axes(),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.swapaxes"
    ),
    native_array=helpers.array_bools(num_arrays=1),
)
def test_numpy_swapaxes(
    dtype_arr_and_axes,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis1, axis2 = dtype_arr_and_axes
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        with_out=False,
        fw=fw,
        frontend="numpy",
        fn_tree="swapaxes",
        a=np.asarray(x, dtype=input_dtype),
        axis1=axis1,
        axis2=axis2,
    )
