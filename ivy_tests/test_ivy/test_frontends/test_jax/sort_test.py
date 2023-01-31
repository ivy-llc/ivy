import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.sort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_sort(
    *,
    dtype_x_axis,
    frontend,
    fn_tree,
    on_device,
    test_flags,
):
    input_dtype, x= dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        with_out=False,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )
