# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# avg_pool1d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.avg_pool1d",
    x_k_s_p_df=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=4,
    ),
    test_with_out=st.just(False),
)
def test_paddle_avg_pool1d(*, x_k_s_p_df, frontend, test_flags, on_device, fn_tree):
    (input_dtype, x, kernel_size, stride, padding) = x_k_s_p_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        exclusive=False,
        ceil_mode=False,
    )
