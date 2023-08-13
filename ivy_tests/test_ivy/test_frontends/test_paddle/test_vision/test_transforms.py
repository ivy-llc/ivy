# global

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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

@handle_frontend_test(
    fn_tree="paddle.vision.transforms.crop",
    dtype_and_x_and_top_and_left_height_width=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1
    ),
)
def test_paddle_crop(
    *,
    dtype_and_x_and_top_and_left_and_height_and_width,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x_and_top_and_left_height_width = dtype_and_x_and_top_and_left_and_height_and_width
    x, top, left, height, width = x_and_top_and_left_height_width
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        img=x[0],
        top=top,
        left=left,
        height=height,
        width=width,
        backend_to_test=backend_fw,
    )
