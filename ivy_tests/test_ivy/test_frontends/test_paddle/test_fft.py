# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.fft.fft",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
    ),
    n=st.one_of(
        st.integers(min_value=2, max_value=10),
        st.just(None),
    ),
    norm=st.sampled_from(["backward", "ortho", "forward"]),
)
def test_paddle_fft(
    dtype_x_axis,
    n,
    norm,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        n=n,
        axis=axis,
        norm=norm,
    )


@handle_frontend_test(
    fn_tree="paddle.fft.fftshift",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_paddle_fttshift(dtype_x_axis, frontend, test_flags, fn_tree, on_device):
    input_dtype, x, axes = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        x=x[0],
        axes=axes,
    )


@handle_frontend_test(
    fn_tree="paddle.fft.ifft",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
    ),
    n=st.one_of(
        st.integers(min_value=2, max_value=10),
        st.just(None),
    ),
    norm=st.sampled_from(["backward", "ortho", "forward"]),
)
def test_paddle_ifft(
    dtype_x_axis,
    n,
    norm,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
        n=n,
        axis=axis,
        norm=norm,
    )
