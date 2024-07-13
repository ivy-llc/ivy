# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_nn.test_layers import (
    _x_and_ifftn_jax,
)


# fft
@handle_frontend_test(
    fn_tree="jax.numpy.fft.fft",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("complex"),
        num_arrays=1,
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        allow_inf=False,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        valid_axis=True,
        force_int_axis=True,
    ),
    n=st.integers(min_value=2, max_value=10),
    norm=st.sampled_from(["backward", "ortho", "forward", None]),
)
def test_jax_numpy_fft(
    dtype_values_axis, n, norm, frontend, backend_fw, test_flags, fn_tree, on_device
):
    dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=values[0],
        n=n,
        axis=axis,
        norm=norm,
        atol=1e-02,
        rtol=1e-02,
    )


# fft2
@handle_frontend_test(
    fn_tree="jax.numpy.fft.fft2",
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex"),
        num_arrays=1,
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        allow_inf=False,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    axes=st.sampled_from([(0, 1), (-1, -2), (1, 0)]),
    s=st.tuples(
        st.integers(min_value=2, max_value=256), st.integers(min_value=2, max_value=256)
    ),
    norm=st.sampled_from(["backward", "ortho", "forward", None]),
)
def test_jax_numpy_fft2(
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
        atol=1e-02,
        rtol=1e-02,
    )


# fftfreq
@handle_frontend_test(
    fn_tree="jax.numpy.fft.fftfreq",
    n=st.integers(min_value=10, max_value=100),
    sample_rate=st.integers(min_value=1, max_value=10),
    dtype=st.one_of(helpers.get_dtypes("float", full=False), st.none()),
)
def test_jax_numpy_fftfreq(
    n, sample_rate, dtype, backend_fw, frontend, test_flags, fn_tree, on_device
):
    d = 1 / sample_rate
    dtype = dtype[0] if dtype is not None else None
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
        dtype=dtype,
    )


# fftshift
@handle_frontend_test(
    fn_tree="jax.numpy.fft.fftshift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), shape=(4,), array_api_dtypes=True
    ),
)
def test_jax_numpy_fftshift(
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


# ifft
@handle_frontend_test(
    fn_tree="jax.numpy.fft.ifft",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("complex"),
        num_arrays=1,
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        allow_inf=False,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        valid_axis=True,
        force_int_axis=True,
    ),
    n=st.integers(min_value=2, max_value=10),
    norm=st.sampled_from(["backward", "ortho", "forward", None]),
)
def test_jax_numpy_ifft(
    dtype_values_axis, n, norm, frontend, backend_fw, test_flags, fn_tree, on_device
):
    dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=values[0],
        n=n,
        axis=axis,
        norm=norm,
        atol=1e-02,
        rtol=1e-02,
    )


# ifft2
@handle_frontend_test(
    fn_tree="jax.numpy.fft.ifft2",
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        allow_inf=False,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    axes=st.sampled_from([(0, 1), (-1, -2), (1, 0)]),
    s=st.tuples(
        st.integers(min_value=2, max_value=256), st.integers(min_value=2, max_value=256)
    ),
    norm=st.sampled_from(["backward", "ortho", "forward", None]),
)
def test_jax_numpy_ifft2(
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
        atol=1e-02,
        rtol=1e-02,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.fft.ifftn",
    dtype_and_x=_x_and_ifftn_jax(),
)
def test_jax_numpy_ifftn(
    dtype_and_x, backend_fw, frontend, test_flags, fn_tree, on_device
):
    input_dtype, x, s, axes, norm = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        atol=1e-09,
        rtol=1e-08,
        a=x,
        s=s,
        axes=axes,
        norm=norm,
    )


# ifftshift
@handle_frontend_test(
    fn_tree="jax.numpy.fft.ifftshift",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        allow_inf=False,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_jax_numpy_ifftshift(
    dtype_values_axis, backend_fw, frontend, test_flags, fn_tree, on_device
):
    dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        x=values[0],
        axes=axis,
    )


# irfftn
@handle_frontend_test(
    fn_tree="jax.numpy.fft.irfftn",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=10,
        valid_axis=True,
        force_int_axis=True,
        num_arrays=1,
    ),
    norm=st.sampled_from(["backward", "ortho", "forward"]),
)
def test_jax_numpy_irfftn(
    dtype_x_axis,
    norm,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    input_dtypes, x, _ = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        atol=1e-3,
        a=x[0],
        s=None,  # TODO: also test cases where `s` and `axes` are not None
        axes=None,
        norm=norm,
    )


# rfft
@handle_frontend_test(
    fn_tree="jax.numpy.fft.rfft",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_value=-1e5,
        max_value=1e5,
        min_num_dims=1,
        min_dim_size=2,
        allow_inf=False,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        valid_axis=True,
        force_int_axis=True,
    ),
    n=st.one_of(
        st.integers(min_value=2, max_value=10),
        st.just(None),
    ),
    norm=st.sampled_from(["backward", "ortho", "forward", None]),
)
def test_jax_numpy_rfft(
    dtype_input_axis, n, norm, frontend, backend_fw, test_flags, fn_tree, on_device
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        n=n,
        axis=axis,
        norm=norm,
        atol=1e-04,
        rtol=1e-04,
    )
