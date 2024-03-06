# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_nn.test_layers import (
    _x_and_ifft,
    _x_and_rfftn,
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
    fn_tree="numpy.fft.fftn",
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=3,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
    axes=st.sampled_from([(0, 1), (-1, -2), (1, 0), None]),
    s=st.tuples(
        st.integers(min_value=2, max_value=256), st.integers(min_value=2, max_value=256)
    )
    | st.none(),
    norm=st.sampled_from(["backward", "ortho", "forward", None]),
)
def test_numpy_fftn(
    dtype_values,
    s,
    axes,
    norm,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, values = dtype_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=values[0],
        s=s,
        axes=axes,
        norm=norm,
    )


@handle_frontend_test(
    fn_tree="numpy.fft.fftshift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(4,), array_api_dtypes=True
    ),
)
def test_numpy_fftshift(
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


# ivy_tests/test_ivy/test_functional/test_experimental/test_nn/test_layers.py


@handle_frontend_test(
    fn_tree="numpy.fft.ifft",
    dtype_and_x=_x_and_ifft(),
)
def test_numpy_ifft(dtype_and_x, backend_fw, frontend, test_flags, fn_tree, on_device):
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
    fn_tree="numpy.fft.ifft2",
    dtype_and_x=_x_and_ifft(),
)
def test_numpy_ifft2(dtype_and_x, backend_fw, frontend, test_flags, fn_tree, on_device):
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
    fn_tree="numpy.fft.ifftn",
    dtype_and_x=_x_and_ifft(),
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
    fn_tree="numpy.fft.ifftshift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(4,), array_api_dtypes=True
    ),
)
def test_numpy_ifftshift(
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
    fn_tree="numpy.fft.rfftn",
    dtype_and_x=_x_and_rfftn(),
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
