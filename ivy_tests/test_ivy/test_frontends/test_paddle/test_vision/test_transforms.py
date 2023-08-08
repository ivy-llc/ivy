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


# hflip
@handle_frontend_test(
    fn_tree="paddle.vision.transforms.hflip",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "int32", "float64", "int64"],
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
