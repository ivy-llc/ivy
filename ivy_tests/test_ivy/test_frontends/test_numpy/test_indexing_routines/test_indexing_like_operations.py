# Testing Function
# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy


@handle_frontend_test(
    fn_tree="numpy.diagonal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        shared_dtype=True,
        min_num_dims=2,
    ),
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
        allow_neg=True,
        max_size=2,
        min_size=2,
        force_tuple=True,
        force_int=False,
    ),
    offset=st.integers(min_value=-1, max_value=1),
)
def test_numpy_diagonal(
    dtype_and_x,
    dtype_and_axis,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    offset,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    axis = dtype_and_axis
    as_variable = as_variable

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        a=ivy.native_array(x, dtype=ivy.int32),
        offset=offset,
        axis1=axis[0],
        axis2=axis[1],
    )
