# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# transpose
@handle_frontend_test(
    fn_tree="numpy.transpose",
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=10,
    ),
    test_with_out=st.just(False),
)
def test_numpy_transpose(
    *,
    array_and_axes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=array,
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


@handle_frontend_test(
    fn_tree="numpy.swapaxes",
    dtype_arr_and_axes=st_dtype_arr_and_axes(),
    test_with_out=st.just(False),
)
def test_numpy_swapaxes(
    *,
    dtype_arr_and_axes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis1, axis2 = dtype_arr_and_axes
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis1=axis1,
        axis2=axis2,
    )
