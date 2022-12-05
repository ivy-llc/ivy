# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            ),
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


# reshape
@handle_frontend_test(
    fn_tree="numpy.reshape",
    dtypes_x_shape=dtypes_x_reshape(),
    order=st.sampled_from(["C", "F", "A"]),
)
def test_numpy_reshape(
    *,
    dtypes_x_shape,
    order,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, x, shape = dtypes_x_shape
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        newshape=shape,
        order=order,
    )


@handle_frontend_test(
    fn_tree="numpy.broadcast_to",
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), ret_shape=True
    ),
    factor=helpers.ints(min_value=1, max_value=5),
)
def test_numpy_broadcast_to(
    *,
    dtype_x_shape,
    factor,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x, shape = dtype_x_shape
    broadcast_shape = (factor,) + shape
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        array=x[0],
        shape=broadcast_shape,
    )


@handle_frontend_test(
    fn_tree="numpy.ravel",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    order=st.sampled_from(["C", "F", "A", "K"]),
)
def test_numpy_ravel(
    *,
    dtype_and_x,
    order,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        order=order,
    )


# moveaxis
@handle_frontend_test(
    fn_tree="numpy.moveaxis",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
    ),
    source=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
    destination=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
)
def test_numpy_moveaxis(
    *,
    dtype_and_a,
    source,
    destination,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
        source=source,
        destination=destination,
    )
