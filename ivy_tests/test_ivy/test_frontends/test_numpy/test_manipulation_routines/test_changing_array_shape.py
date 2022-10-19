# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            )
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


# reshape
@handle_cmd_line_args
@given(
    dtypes_x_shape=dtypes_x_reshape(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.reshape"
    ),
)
def test_numpy_reshape(
    dtypes_x_shape,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
):
    dtypes, x, shape = dtypes_x_shape
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="reshape",
        x=x[0],
        newshape=shape,
    )


@handle_cmd_line_args
@given(
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), ret_shape=True
    ),
    factor=helpers.ints(min_value=1, max_value=5),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.broadcast_to"
    ),
)
def test_numpy_broadcast_to(
    dtype_x_shape,
    factor,
    as_variable,
    native_array,
    num_positional_args,
):
    dtype, x, shape = dtype_x_shape
    broadcast_shape = (factor,) + shape
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        frontend="numpy",
        fn_tree="broadcast_to",
        array=x[0],
        shape=broadcast_shape,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ravel"
    ),
)
def test_numpy_ravel(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        frontend="numpy",
        fn_tree="ravel",
        a=x[0],
    )
