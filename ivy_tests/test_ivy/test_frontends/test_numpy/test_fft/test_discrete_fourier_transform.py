# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_nn.test_layers import (
    x_and_ifft,
    x_and_rfftn,
)

# ivy_tests/test_ivy/test_functional/test_experimental/test_nn/test_layers.py


@handle_frontend_test(
    fn_tree="numpy.fft.ifft",
    dtype_and_x=x_and_ifft(),
)
def test_numpy_iftt(dtype_and_x, backend_fw, frontend, test_flags, fn_tree, on_device):
    input_dtype, x, dim, norm, n = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x,
        n=n,
        axis=dim,
        norm=norm,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.ifftshift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(4,), array_api_dtypes=True
    ),
)
def test_numpy_ifttshift(
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


@handle_frontend_test(
    fn_tree="numpy.fft.fft",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        shape=(2,),
        min_axis=-1,
        force_int_axis=True,
    ),
    norm=st.sampled_from(["backward", "ortho", "forward"]),
    n=st.integers(min_value=2, max_value=10),
)
def test_numpy_fft(
    dtype_input_axis, norm, n, backend_fw, frontend, test_flags, fn_tree, on_device
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x[0],
        n=n,
        axis=axis,
        norm=norm,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.fftshift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(4,), array_api_dtypes=True
    ),
)
def test_numpy_fttshift(
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


@handle_frontend_test(
    fn_tree="numpy.fft.rfft",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        shape=(2,),
        min_axis=-1,
        force_int_axis=True,
    ),
    norm=st.sampled_from(["backward", "ortho", "forward"]),
    n=st.integers(min_value=2, max_value=5),
)
def test_numpy_rfft(
    dtype_input_axis, norm, n, backend_fw, frontend, test_flags, fn_tree, on_device
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x[0],
        n=n,
        axis=axis,
        norm=norm,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.ihfft",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        shape=(2,),
        min_axis=-1,
        force_int_axis=True,
    ),
    norm=st.sampled_from(["backward", "ortho", "forward"]),
    n=st.integers(min_value=2, max_value=5),
)
def test_numpy_ihfft(
    dtype_input_axis, norm, n, backend_fw, frontend, test_flags, fn_tree, on_device
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x[0],
        n=n,
        axis=axis,
        norm=norm,
    )



@handle_frontend_test(
    fn_tree="numpy.fft.fftfreq",
    n=st.integers(min_value=10, max_value=100),
    sample_rate=st.integers(min_value=1, max_value=10),
)
def test_numpy_fftfreq(
    n, sample_rate, backend_fw, frontend, test_flags, fn_tree, on_device
):
    d = 1 / sample_rate
    helpers.test_frontend_function(
        input_dtypes=[int],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        n=n,
        d=d,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.rfftfreq",
    n=st.integers(min_value=10, max_value=100),
    sample_rate=st.integers(min_value=1, max_value=10),
)
def test_numpy_rfftfreq(
    n, sample_rate, backend_fw, frontend, test_flags, fn_tree, on_device
):
    d = 1 / sample_rate
    helpers.test_frontend_function(
        input_dtypes=[int],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        n=n,
        d=d,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.ifftn",
    dtype_and_x=x_and_ifft(),
)
def test_numpy_ifftn(dtype_and_x, backend_fw, frontend, test_flags, fn_tree, on_device):
    input_dtype, x, dim, norm, n = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x,
        s=None,
        axes=None,
        norm=norm,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.rfftn",
    dtype_and_x=x_and_rfftn(),
)
def test_numpy_rfftn(dtype_and_x, frontend, backend_fw, test_flags, fn_tree, on_device):
    dtype, x, s, axes, norm = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x,
        s=s,
        axes=axes,
        norm=norm,
    )


