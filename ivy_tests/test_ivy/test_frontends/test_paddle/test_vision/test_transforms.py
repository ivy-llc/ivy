# global

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

from hypothesis import strategies as st


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


@st.composite
def _pad_helper(draw):
    input_dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_value=0,
            min_value=0,
            min_num_dims=3,
            max_num_dims=3,
            min_dim_size=3,
            max_dim_size=3,
        ),
    )
    pad = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_value=0,
            min_dim_size=1,
            min_num_dims=x[0].ndim,
            max_num_dims=x[0].ndim,
        ),
    )
    # mode = st.one_of([1,2,3])
    # value = st.floats()
    return input_dtypes, x[0], pad


# pad
@handle_frontend_test(
    fn_tree="paddle.vision.transforms.pad", dtypes_x_pad=_pad_helper()
)
def test_paddle_pad(
    *,
    dtypes_x_pad,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, pad = dtypes_x_pad
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        backend_to_test=backend_fw,
        x=x,
        pad=pad,
    )
