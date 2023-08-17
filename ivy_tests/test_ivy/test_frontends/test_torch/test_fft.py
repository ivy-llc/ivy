from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test


@st.composite
def x_and_rfftn(draw):
    min_rfftn_points = 2
    dtype = draw(helpers.get_dtypes("float"))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=3
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e10,
            max_value=1e10,
            large_abs_safety_factor=2.5,
            small_abs_safety_factor=2.5,
            safety_factor_scale="log",
        )
    )
    axes = draw(
        st.lists(
            st.integers(0, len(x_dim) - 1), min_size=1, max_size=len(x_dim), unique=True
        )
    )
    s = draw(
        st.lists(
            st.integers(min_rfftn_points, 256), min_size=len(axes), max_size=len(axes)
        )
    )
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    return dtype, x, s, axes, norm


@handle_frontend_test(
    fn_tree="torch.fft.rfftn",
    d_x_d_s_n=x_and_rfftn(),
    test_with_out=st.just(False),
)
def test_torch_rfftn(
    *,
    d_x_d_s_n,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x, s, axes, norm = d_x_d_s_n
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        input=x,
        s=s,
        dim=axes,
        norm=norm,
    )
