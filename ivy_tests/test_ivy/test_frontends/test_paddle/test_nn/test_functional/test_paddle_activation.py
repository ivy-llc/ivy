# global

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# selu
@handle_frontend_test(
    fn_tree="paddle.nn.functional.selu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        safety_factor_scale="log",
        small_abs_safety_factor=20,
    ),
    scale=helpers.ints(min_value=2, max_value=10),
    alpha=helpers.ints(min_value=1, max_value=10),
)
def test_paddle_selu(
    *,
    dtype_and_x,
    scale,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        alpha=alpha,
        scale=scale,
    )
