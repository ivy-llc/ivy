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
    fn_tree="paddle.vision.transforms.adjust_hue",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=3,
        max_dim_size=3,
        min_value=0,
    ),
    hue_factor=helpers.floats(min_value=-0.5, max_value=0.5, abs_smallest_val=1e-6),
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
        on_device=on_device,
        img=x[0],
        hue_factor=hue_factor,
    )
