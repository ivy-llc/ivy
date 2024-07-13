# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_nn.test_functional.test_linear_functions import (  # noqa: E501
    _x_and_linear,
)


# --- Helpers --- #
# --------------- #


# interpolate
@st.composite
def _interp_args(draw, mode=None, mode_list=None):
    mixed_fn_compos = draw(st.booleans())
    curr_backend = ivy.current_backend_str()
    torch_modes = [
        "linear",
        "bilinear",
        "trilinear",
        "nearest",
        "nearest-exact",
        "area",
    ]

    tf_modes = [
        "linear",
        "bilinear",
        "trilinear",
        "nearest-exact",
        "tf_area",
        "tf_bicubic",
        "lanczos3",
        "lanczos5",
        "mitchellcubic",
        "gaussian",
    ]

    jax_modes = [
        "linear",
        "bilinear",
        "trilinear",
        "nearest-exact",
        "tf_bicubic",
        "lanczos3",
        "lanczos5",
    ]

    if not mode and not mode_list:
        if curr_backend == "torch" and not mixed_fn_compos:
            mode = draw(st.sampled_from(torch_modes))
        elif curr_backend == "tensorflow" and not mixed_fn_compos:
            mode = draw(st.sampled_from(tf_modes))
        elif curr_backend == "jax" and not mixed_fn_compos:
            mode = draw(st.sampled_from(jax_modes))
        else:
            mode = draw(
                st.sampled_from(
                    [
                        "linear",
                        "bilinear",
                        "trilinear",
                        "nearest",
                        "nearest-exact",
                        "area",
                        "tf_area",
                        "tf_bicubic",
                        "lanczos3",
                        "lanczos5",
                        "mitchellcubic",
                        "gaussian",
                    ]
                )
            )
    elif mode_list:
        mode = draw(st.sampled_from(mode_list))
    align_corners = draw(st.booleans())
    if curr_backend in ["tensorflow", "jax"] and not mixed_fn_compos:
        align_corners = False
    if mode == "linear":
        num_dims = 3
    elif mode in [
        "bilinear",
        "tf_bicubic",
        "bicubic",
        "mitchellcubic",
        "gaussian",
    ]:
        num_dims = 4
    elif mode == "trilinear":
        num_dims = 5
    elif mode in [
        "nearest",
        "area",
        "tf_area",
        "lanczos3",
        "lanczos5",
        "nearest-exact",
    ]:
        num_dims = (
            draw(
                helpers.ints(min_value=1, max_value=3, mixed_fn_compos=mixed_fn_compos)
            )
            + 2
        )
        align_corners = False
    if curr_backend == "tensorflow" and not mixed_fn_compos:
        num_dims = 3
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(
                "float", mixed_fn_compos=mixed_fn_compos
            ),
            min_num_dims=num_dims,
            max_num_dims=num_dims,
            min_dim_size=2,
            max_dim_size=5,
            large_abs_safety_factor=50,
            small_abs_safety_factor=50,
            safety_factor_scale="log",
        )
    )
    if draw(st.booleans()):
        scale_factor = draw(
            st.one_of(
                helpers.lists(
                    x=helpers.floats(
                        min_value=1.0, max_value=2.0, mixed_fn_compos=mixed_fn_compos
                    ),
                    min_size=num_dims - 2,
                    max_size=num_dims - 2,
                ),
                helpers.floats(
                    min_value=1.0, max_value=2.0, mixed_fn_compos=mixed_fn_compos
                ),
            )
        )
        recompute_scale_factor = draw(st.booleans())
        size = None
    else:
        size = draw(
            st.one_of(
                helpers.lists(
                    x=helpers.ints(
                        min_value=1, max_value=3, mixed_fn_compos=mixed_fn_compos
                    ),
                    min_size=num_dims - 2,
                    max_size=num_dims - 2,
                ),
                st.integers(min_value=1, max_value=3),
            )
        )
        recompute_scale_factor = False
        scale_factor = None
    if curr_backend in ["tensorflow", "jax"] and not mixed_fn_compos:
        if not recompute_scale_factor:
            recompute_scale_factor = True

    return (dtype, x, mode, size, align_corners, scale_factor, recompute_scale_factor)


# zeropad2d
@st.composite
def _zero2pad(draw):
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            ret_shape=True,
            min_num_dims=4,
            max_num_dims=4,
            min_value=-100,
            max_value=100,
        )
    )
    ndim = len(shape)
    min_dim = min(shape)
    padding = draw(
        st.lists(
            st.integers(min_value=0, max_value=min_dim),
            min_size=ndim,
            max_size=ndim,
        )
    )
    return dtype, input, padding


@st.composite
def paddle_unfold_handler(draw, dtype):
    dtype = draw(dtype)
    h_size = draw(helpers.ints(min_value=10, max_value=30))
    w_size = draw(helpers.ints(min_value=10, max_value=30))
    channels = draw(helpers.ints(min_value=1, max_value=3))
    batch = draw(helpers.ints(min_value=1, max_value=10))

    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=[batch, channels, h_size, w_size],
            min_value=0,
            max_value=1,
        )
    )

    kernel_sizes = draw(helpers.ints(min_value=1, max_value=3))
    strides = draw(helpers.ints(min_value=1, max_value=3))
    paddings = draw(helpers.ints(min_value=1, max_value=3))
    dilations = draw(helpers.ints(min_value=1, max_value=3))
    return dtype, x, kernel_sizes, strides, paddings, dilations


# --- Main --- #
# ------------ #


# Cosine Similarity
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.cosine_similarity",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=2,
        max_value=5,
        min_dim_size=2,
        shape=(4, 4),
    ),
    axis=st.integers(min_value=-1, max_value=1),
)
def test_paddle_cosine_similarity(
    *,
    d_type_and_x,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        x1=x[0],
        x2=x[1],
        axis=axis,
    )


# dropout
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.dropout",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        shared_dtype=True,
        min_value=2,
        max_value=5,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    p=st.floats(min_value=0.0, max_value=1.0),
    axis=st.integers(min_value=0, max_value=1),
    training=st.booleans(),
    mode=st.sampled_from(["upscale_in_train", "downscale_in_infer"]),
)
def test_paddle_dropout(
    *,
    d_type_and_x,
    p,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
    training,
    axis,
    mode,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        p=p,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        test_flags=test_flags,
        on_device=on_device,
        x=x[0],
        training=training,
        axis=axis,
        mode=mode,
    )


# Dropout2d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.dropout2d",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
        shared_dtype=True,
        min_value=2,
        max_value=5,
        min_dim_size=4,
        shape=(
            st.integers(min_value=2, max_value=10),
            4,
            st.integers(min_value=12, max_value=64),
            st.integers(min_value=12, max_value=64),
        ),
    ),
    p=st.floats(min_value=0.0, max_value=1.0),
    training=st.booleans(),
    data_format=st.sampled_from(["NCHW", "NHWC"]),
)
def test_paddle_dropout2d(
    *,
    d_type_and_x,
    p,
    training,
    data_format,
    backend_fw,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        p=p,
        training=training,
        data_format=data_format,
    )


# Dropout3d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.dropout3d",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=5,
        max_num_dims=5,
    ),
    p=st.floats(min_value=0.0, max_value=1.0),
    training=st.booleans(),
    data_format=st.sampled_from(["NCDHW", "NDHWC"]),
)
def test_paddle_dropout3d(
    *,
    d_type_and_x,
    p,
    training,
    data_format,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        p=p,
        training=training,
        data_format=data_format,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.interpolate",
    dtype_x_mode=_interp_args(),
)
def test_paddle_interpolate(
    dtype_x_mode,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    (
        input_dtype,
        x,
        mode,
        size,
        align_corners,
        scale_factor,
        recompute_scale_factor,
    ) = dtype_x_mode

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


# linear
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.linear",
    dtype_x_weight_bias=_x_and_linear(
        dtypes=helpers.get_dtypes("valid", full=False),
    ),
)
def test_paddle_linear(
    *,
    dtype_x_weight_bias,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    dtype, x, weight, bias = dtype_x_weight_bias
    weight = ivy.swapaxes(weight, -1, -2)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        weight=weight,
        bias=bias,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.unfold",
    dtype_inputs=paddle_unfold_handler(dtype=helpers.get_dtypes("valid", full=False)),
)
def test_paddle_unfold(
    *,
    dtype_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, kernel_sizes, strides, paddings, dilations = dtype_inputs
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        dilations=dilations,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.zeropad2d",
    d_type_and_x_paddings=_zero2pad(),
    dataformat=st.sampled_from(["NCHW", "NHWC"]),
)
def test_paddle_zeropad2d(
    *,
    d_type_and_x_paddings,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    dataformat,
):
    dtype, x, padding = d_type_and_x_paddings
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        padding=padding,
        data_format=dataformat,
    )
