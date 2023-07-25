# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# fft
@handle_frontend_test(
    fn_tree="jax.numpy.fft.fft",
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
        test_values=True,
        a=values[0],
        n=n,
        axis=axis,
        norm=norm,
        atol=1e-02,
        rtol=1e-02,
    )
