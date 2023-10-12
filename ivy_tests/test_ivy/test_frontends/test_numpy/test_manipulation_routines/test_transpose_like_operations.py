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
    backend_fw,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=array,
        axes=axes,
    )


# swapaxes
@handle_frontend_test(
    fn_tree="numpy.swapaxes",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    axis1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"), force_int=True
    ),
    axis2=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"), force_int=True
    ),
    test_with_out=st.just(False),
)
def test_numpy_swapaxes(
    *,
    dtype_and_x,
    axis1,
    axis2,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis1=axis1,
        axis2=axis2,
    )


# rollaxis
@handle_frontend_test(
    fn_tree="numpy.rollaxis",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=3,
        min_dim_size=2,
    ),
    axis=helpers.ints(min_value=-2, max_value=2),
    start=helpers.ints(min_value=-2, max_value=2),
    test_with_out=st.just(False),
)
def test_numpy_rollaxis(
    *,
    dtype_and_a,
    axis,
    start,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
        axis=axis,
        start=start,
    )
