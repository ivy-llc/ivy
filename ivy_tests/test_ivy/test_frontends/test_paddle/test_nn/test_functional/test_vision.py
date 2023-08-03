# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _affine_grid_helper(draw):
    align_corners = draw(st.booleans())
    dims = draw(st.integers(4, 5))
    if dims == 4:
        size = draw(
            st.tuples(
                st.integers(1, 20),
                st.integers(1, 20),
                st.integers(2, 20),
                st.integers(2, 20),
            )
        )
        theta_dtype, theta = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                min_value=0,
                max_value=1,
                shape=(size[0], 2, 3),
            )
        )
        return theta_dtype, theta[0], size, align_corners
    else:
        size = draw(
            st.tuples(
                st.integers(1, 20),
                st.integers(1, 20),
                st.integers(2, 20),
                st.integers(2, 20),
                st.integers(2, 20),
            )
        )
        theta_dtype, theta = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                min_value=0,
                max_value=1,
                shape=(size[0], 3, 4),
            )
        )
        return theta_dtype, theta[0], size, align_corners


@handle_frontend_test(
    fn_tree="paddle.nn.functional.affine_grid",
    dtype_and_input_and_other=_affine_grid_helper(),
)
def test_paddle_affine_grid(
    *, dtype_and_input_and_other, on_device, backend_fw, fn_tree, frontend, test_flags
):
    dtype, theta, size, align_corners = dtype_and_input_and_other

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        theta=theta,
        out_shape=size,
        align_corners=align_corners,
    )
