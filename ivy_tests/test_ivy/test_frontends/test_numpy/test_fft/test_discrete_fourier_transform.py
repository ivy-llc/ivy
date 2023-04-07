import ivy
import ivy.numpy as np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.fft.ifft",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), 
        shape=(4,), array_api_dtypes=True
    )
)    
def test_numpy_iftt(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device
):
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
        norm=None
    )


@handle_frontend_test(
    fn_tree="numpy.fft.ifft2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        available_shapes=[(4, 4), (3, 5), (2, 2, 2)]
    )
)
def test_numpy_ifft2(
        dtype_and_x,
        frontend,
        test_flags,
        fn_tree,
        on_device
):
    input_dtype, x = dtype_and_x
    a = ivy.array(x, dtype=ivy.complex128)

    # Test `ifft2` function with different parameter values
    for n in [None, 4]:
        for axis in [(-1, -2), (-2, -1)]:
            for norm in [None, "forward", "backward"]:

                np_output = np.fft.ifft2(x, n=n, axes=axis, norm=norm)

                ivy_output = ivy.to_numpy(ivy.ifft2(a, n=n, axis=axis, norm=norm))

                helpers.assert_allclose(np_output, ivy_output, rtol=1e-4, atol=1e-4, backend=frontend)

