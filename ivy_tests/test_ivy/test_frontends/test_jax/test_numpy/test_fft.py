# global

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="jax.numpy.fft.fftshift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), shape=(4,), array_api_dtypes=True
    ),
)
def test_jax_numpy_fttshift(
    dtype_and_x, backend_fw, frontend, test_flags, fn_tree, on_device
):
    input_dtype, arr = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        x=arr[0],
        axes=None,
    )
