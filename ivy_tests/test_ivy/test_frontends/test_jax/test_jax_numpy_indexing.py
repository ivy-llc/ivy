# global
from hypothesis import strategies as st, assume
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# diagonal
@st.composite
def dims_and_offset(draw, shape):
    shape_actual = draw(shape)
    dim1 = draw(helpers.get_axis(shape=shape, force_int=True))
    dim2 = draw(helpers.get_axis(shape=shape, force_int=True))
    offset = draw(
        st.integers(min_value=-shape_actual[dim1], max_value=shape_actual[dim1])
    )
    return dim1, dim2, offset


@handle_frontend_test(
    fn_tree="jax.numpy.diagonal",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dims_and_offset=dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
)
def test_jax_numpy_diagonal(
    *,
    dtype_and_values,
    dims_and_offset,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    axis1, axis2, offset = dims_and_offset
    a = value[0]
    num_of_dims = len(np.shape(a))
    assume(axis1 != axis2)
    if axis1 < 0:
        assume(axis1 + num_of_dims != axis2)
    if axis2 < 0:
        assume(axis1 != axis2 + num_of_dims)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


# diag
@st.composite
def _diag_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            small_abs_safety_factor=2,
            large_abs_safety_factor=2,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=2,
            min_dim_size=1,
            max_dim_size=50,
        )
    )
    shape = x[0].shape
    if len(shape) == 2:
        k = draw(helpers.ints(min_value=-shape[0] + 1, max_value=shape[1] - 1))
    else:
        k = draw(helpers.ints(min_value=0, max_value=shape[0]))
    return dtype, x, k


@handle_frontend_test(
    fn_tree="jax.numpy.diag",
    dtype_x_k=_diag_helper(),
)
def test_jax_numpy_diag(
    *,
    dtype_x_k,
    num_positional_args,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x, k = dtype_x_k
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        v=x[0],
        k=k,
    )
