# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _chw_image_shape_helper(draw):
    c = draw(st.sampled_from([1, 3]), label="channel")
    h = draw(helpers.ints(min_value=1, max_value=100), label="height")
    w = draw(helpers.ints(min_value=1, max_value=100), label="width")

    shape = (c, h, w)
    return shape


# --- Main --- #
# ------------ #


# adjust_brightness
@handle_frontend_test(
    fn_tree="paddle.vision.transforms.adjust_brightness",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=_chw_image_shape_helper(),
    ),
    brightness_factor=helpers.floats(min_value=0),
)
def test_paddle_adjust_brightness(
    *,
    dtype_and_x,
    brightness_factor,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        img=x[0],
        brightness_factor=brightness_factor,
    )


# adjust_hue
@handle_frontend_test(
    fn_tree="paddle.vision.transforms.adjust_hue",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=0,
        max_value=255,
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=3,
        max_dim_size=3,
    ),
    hue_factor=helpers.floats(min_value=-0.5, max_value=0.5),
)
def test_paddle_adjust_hue(
    *,
    dtype_and_x,
    hue_factor,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-3,
        atol=1e-3,
        on_device=on_device,
        img=x[0],
        hue_factor=hue_factor,
    )


# hflip
@handle_frontend_test(
    fn_tree="paddle.vision.transforms.hflip",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=0,
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=3,
        max_dim_size=3,
    ),
)
def test_paddle_hflip(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        img=x[0],
    )


# pad
@handle_frontend_test(
    fn_tree="paddle.vision.transforms.pad",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=3,
        max_dim_size=5,
    ),
    padding=st.one_of(
        st.integers(min_value=0, max_value=2),
        st.tuples(
            st.integers(min_value=0, max_value=2), st.integers(min_value=0, max_value=2)
        ),
        st.tuples(
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=2),
        ),
    ),
    fill=st.integers(min_value=-5, max_value=5),
    padding_mode=st.sampled_from(["constant", "edge", "reflect"]),
)
def test_paddle_pad(
    *,
    dtype_and_x,
    padding,
    fill,
    padding_mode,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        img=x[0],
        padding=padding,
        fill=fill,
        padding_mode=padding_mode,
        backend_to_test=backend_fw,
    )


# to_tensor
@handle_frontend_test(
    fn_tree="paddle.to_tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_paddle_to_tensor(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        pic=input[0],
    )


@handle_frontend_test(
    fn_tree="paddle.vision.transforms.vflip",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=4,
    ),
)
def test_paddle_vflip(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        img=x[0],
        backend_to_test=backend_fw,
    )
