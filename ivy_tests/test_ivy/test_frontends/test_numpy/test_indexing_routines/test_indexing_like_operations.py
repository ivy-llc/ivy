# Testing Function
# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.diagonal",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_axes_size=2,
        max_axes_size=2,
        valid_axis=True,
    ),
    offset=st.integers(min_value=-1, max_value=1),
)
def test_numpy_diagonal(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    offset,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        a=x[0],
        offset=offset,
        axis1=axis[0],
        axis2=axis[1],
    )


@handle_frontend_test(
    fn_tree="numpy.diag_indices",
    n=helpers.ints(min_value=1, max_value=10),
    ndim=helpers.ints(min_value=2, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_numpy_diag_indices(
    n,
    ndim,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        ndim=ndim,
    )
