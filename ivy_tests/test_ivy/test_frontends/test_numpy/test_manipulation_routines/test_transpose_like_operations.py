# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def st_dtype_arr_and_axes(draw):
    dtype, x = draw(
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
        )
    )
    axis1, axis2 = draw(
        helpers.get_axis(
            shape=np.array(x).shape,
            sorted=False,
            unique=False,
            min_size=2,
            max_size=2,
            force_tuple=True,
        )
    )
    return dtype, x, axis1, axis2


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
        # torch: uint8 is not supported --> set valid_types!!
        axis1=axis1,
        axis2=axis2,
    )
