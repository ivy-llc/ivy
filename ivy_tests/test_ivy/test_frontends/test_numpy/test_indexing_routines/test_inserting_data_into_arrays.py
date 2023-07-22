# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.fill_diagonal",
    dtype_x_axis=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        min_dim_size=2,
        max_num_dims=2,
    ),
    val=helpers.floats(min_value=-10, max_value=10),
    wrap=helpers.get_dtypes(kind="bool"),
    test_with_out=st.just(False),
)
def test_numpy_fill_diagonal(
    dtype_x_axis,
    wrap,
    val,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_x_axis
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        on_device=on_device,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        a=x[0],
        val=val,
        wrap=wrap,
    )
