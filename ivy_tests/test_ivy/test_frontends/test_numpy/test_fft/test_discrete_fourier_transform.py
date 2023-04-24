import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.fft.ifft",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(4,), array_api_dtypes=True
    ),
)
def test_numpy_iftt(dtype_and_x, frontend, test_flags, fn_tree, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x,
        n=None,
        axis=-1,
        norm=None,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.ifftshift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(4,), array_api_dtypes=True
    ),
)
def test_numpy_ifttshift(dtype_and_x, frontend, test_flags, fn_tree, on_device):
    input_dtype, arr = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        x=arr[0],
        axes=None,
    )
