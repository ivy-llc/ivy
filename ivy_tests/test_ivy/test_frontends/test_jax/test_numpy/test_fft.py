# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def dtype_x_axis_norm_n(draw):
    min_fft_points = 2
    dtype = draw(helpers.get_dtypes('valid'))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100,
            min_num_dims=1, max_num_dims=4
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
    norm = draw(st.sampled_from([None, 'backward', 'forward', 'ortho']))
    n = draw(st.one_of(st.integers(min_fft_points, 256), st.none()))
    return dtype, x, axis, norm, n

@handle_frontend_test(
    fn_tree="jax.numpy.fft.ifft",
    dtype_x_axis_norm_n=dtype_x_axis_norm_n(),
)
def test_jax_numpy_iftt(
        dtype_x_axis_norm_n,
        backend_fw,
        frontend,
        test_flags,
        fn_tree,
        on_device):
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
