from hypothesis import assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_nn.test_layers import (
    _interp_args,
)


# --- Helpers --- #
# --------------- #


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


@st.composite
def _pad_generator(draw, shape, mode):
    pad = ()
    m = max(int((len(shape) + 1) / 2), 1)
    for i in range(m):
        if mode != "constant":
            if i < 2:
                max_pad_value = 0
        else:
            max_pad_value = shape[i] - 1
        pad = pad + draw(
            st.tuples(
                st.integers(min_value=-3, max_value=max(0, max_pad_value)),
                st.integers(min_value=-3, max_value=max(0, max_pad_value)),
            )
        )
    return pad


@st.composite
def _pad_helper(draw):
    mode = draw(
        st.sampled_from(
            [
                "constant",
                "reflect",
                "replicate",
                "circular",
            ]
        )
    )
    min_v = 1
    max_v = 5
    if mode != "constant":
        min_v = 3
        if mode == "reflect":
            max_v = 4
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            ret_shape=True,
            min_num_dims=min_v,
            max_num_dims=max_v,
            min_dim_size=5,
            min_value=-1e05,
            max_value=1e05,
        )
    )
    padding = draw(_pad_generator(shape, mode))
    if mode == "constant":
        value = draw(helpers.ints(min_value=0, max_value=4) | st.none())
    else:
        value = 0.0
    return dtype, input[0], padding, value, mode


@st.composite
def grid_sample_helper(draw, dtype, mode, mode_3d, padding_mode):
    dtype = draw(dtype)
    align_corners = draw(st.booleans())
    dims = draw(st.integers(4, 5))
    height = draw(helpers.ints(min_value=5, max_value=10))
    width = draw(helpers.ints(min_value=5, max_value=10))
    channels = draw(helpers.ints(min_value=1, max_value=3))

    grid_h = draw(helpers.ints(min_value=2, max_value=4))
    grid_w = draw(helpers.ints(min_value=2, max_value=4))
    batch = draw(helpers.ints(min_value=1, max_value=5))

    padding_mode = draw(st.sampled_from(padding_mode))
    if dims == 4:
        mode = draw(st.sampled_from(mode))
        x = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=[batch, channels, height, width],
                min_value=-1,
                max_value=1,
            )
        )

        grid = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=[batch, grid_h, grid_w, 2],
                min_value=-1,
                max_value=1,
            )
        )
    elif dims == 5:
        mode = draw(st.sampled_from(mode_3d))
        depth = draw(helpers.ints(min_value=10, max_value=15))
        grid_d = draw(helpers.ints(min_value=5, max_value=10))
        x = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=[batch, channels, depth, height, width],
                min_value=-1,
                max_value=1,
            )
        )

        grid = draw(
            helpers.array_values(
                dtype=dtype[0],
                shape=[batch, grid_d, grid_h, grid_w, 3],
                min_value=-1,
                max_value=1,
            )
        )
    return dtype, x, grid, mode, padding_mode, align_corners


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="torch.nn.functional.affine_grid",
    dtype_and_input_and_other=_affine_grid_helper(),
)
def test_torch_affine_grid(
    *,
    dtype_and_input_and_other,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
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
        size=size,
        align_corners=align_corners,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.grid_sample",
    dtype_x_grid_modes=grid_sample_helper(
        dtype=helpers.get_dtypes("valid", full=False),
        mode=["nearest", "bilinear", "bicubic"],
        mode_3d=["nearest", "bilinear"],
        padding_mode=["border", "zeros", "reflection"],
    ),
)
def test_torch_grid_sample(
    *,
    dtype_x_grid_modes,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, grid, mode, padding_mode, align_corners = dtype_x_grid_modes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.interpolate",
    dtype_and_input_and_other=_interp_args(
        mode_list="torch",
    ),
    number_positional_args=st.just(2),
)
def test_torch_interpolate(
    *,
    dtype_and_input_and_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    (
        input_dtype,
        x,
        mode,
        size,
        align_corners,
        scale_factor,
        recompute_scale_factor,
    ) = dtype_and_input_and_other
    if mode not in ["linear", "bilinear", "bicubic", "trilinear"]:
        align_corners = None
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-03,
        input=x[0],
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.pad",
    dtype_and_input_and_other=_pad_helper(),
)
def test_torch_pad(
    *,
    dtype_and_input_and_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, padding, value, mode = dtype_and_input_and_other
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        pad=padding,
        mode=mode,
        value=value,
    )


# pixel_shuffle
@handle_frontend_test(
    fn_tree="torch.nn.functional.pixel_shuffle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
    ),
    factor=helpers.ints(min_value=1),
)
def test_torch_pixel_shuffle(
    *,
    dtype_and_x,
    factor,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(ivy.shape(x[0])[1] % (factor**2) == 0)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        upscale_factor=factor,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.pixel_unshuffle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
    ),
    factor=helpers.ints(min_value=1),
)
def test_torch_pixel_unshuffle(
    *,
    dtype_and_x,
    factor,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume((ivy.shape(x[0])[2] % factor == 0) & (ivy.shape(x[0])[3] % factor == 0))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        downscale_factor=factor,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.upsample",
    dtype_and_input_and_other=_interp_args(),
    number_positional_args=st.just(2),
)
def test_torch_upsample(
    *,
    dtype_and_input_and_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, mode, size, align_corners = dtype_and_input_and_other
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        size=size,
        mode=mode,
        align_corners=align_corners,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.upsample_bilinear",
    dtype_and_input_and_other=_interp_args(mode="bilinear"),
    number_positional_args=st.just(2),
)
def test_torch_upsample_bilinear(
    *,
    dtype_and_input_and_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, _, size, _, scale_factor, _ = dtype_and_input_and_other
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        size=size,
        scale_factor=scale_factor,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.upsample_nearest",
    dtype_and_input_and_other=_interp_args(mode="nearest"),
    number_positional_args=st.just(2),
)
def test_torch_upsample_nearest(
    *,
    dtype_and_input_and_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, _, size, _ = dtype_and_input_and_other
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        size=size,
    )
