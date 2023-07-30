# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def dtype_x_axis_norm_n(draw):
    min_fft_points = 2
    dtype = draw(helpers.get_dtypes("valid"))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e-10,
            max_value=1e10,
        )
    )
    axis = draw(st.integers(1 - len(list(x_dim)), len(list(x_dim)) - 1))
    norm = draw(st.sampled_from([None, "backward", "forward", "ortho"]))
    n = draw(st.one_of(st.integers(min_fft_points, 256), st.none()))
    return dtype, x, axis, norm, n

@handle_frontend_test(
    fn_tree="jax.numpy.fft.ifft",
    dtype_x_axis_norm_n=dtype_x_axis_norm_n(),
)
def test_jax_numpy_iftt(
    dtype_x_axis_norm_n, backend_fw, frontend, test_flags, fn_tree, on_device
):
    input_dtypes, x, axis, norm, n = dtype_x_axis_norm_n
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        a=x,
        n=n,
        axis=axis,
        norm=norm,
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
