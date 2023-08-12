# global
import ivy
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy.numpy as np
from ivy.functional.frontends.paddle.nn.functional.vision import grid_sample
import numpy


# pixel_shuffle
@handle_frontend_test(
    fn_tree="paddle.nn.functional.pixel_shuffle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        min_value=0,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=3,
    ),
    factor=helpers.ints(min_value=1),
    data_format=st.sampled_from(["NCHW", "NHWC"]),
)
def test_paddle_pixel_shuffle(
    *,
    dtype_and_x,
    factor,
    data_format,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    if data_format == "NCHW":
        assume(ivy.shape(x[0])[1] % (factor**2) == 0)
    else:
        assume(ivy.shape(x[0])[3] % (factor**2) == 0)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        upscale_factor=factor,
        data_format=data_format,
        backend_to_test=backend_fw,
    )


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


def test_grid_sample_nearest():
    input_data = np.array([[[[1, 2], [3, 4]]]], dtype=numpy.float32)
    grid_data = np.array([[[[0.5, 0.5], [1.5, 0.5]]]], dtype=numpy.float32)
    input = ivy.array(input_data)
    grid = ivy.array(grid_data)

    output = grid_sample(input, grid, mode="nearest", padding_mode="zeros")

    expected_output = np.array([[[[2, 3]]]], dtype=np.float32)
    numpy.testing.assert_allclose(output, expected_output, rtol=1e-5)


@handle_frontend_test()
def test_grid_sample_bilinear():
    input_data = np.array([[[[1, 2], [3, 4]]]], dtype=numpy.float32)
    grid_data = np.array([[[[0.5, 0.5], [1.5, 0.5]]]], dtype=numpy.float32)
    input = ivy.array(input_data)
    grid = ivy.array(grid_data)

    output = grid_sample(input, grid, mode="bilinear", padding_mode="zeros")

    expected_output = np.array([[[[1.75, 2.25]]]], dtype=numpy.float32)
    numpy.testing.assert_allclose(output, expected_output, rtol=1e-5)
