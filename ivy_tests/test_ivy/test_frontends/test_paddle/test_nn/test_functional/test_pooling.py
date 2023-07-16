# global
from hypothesis import strategies as st

# local	# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# avg_pool1d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.pooling.avg_pool1d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=3,
    ),
    ceil_mode=st.booleans(),
    data_format=st.sampled_from(["NCW", "NWC"]),
)
def test_paddle_avg_pool1d(
    dtype_x_k_s,
    ceil_mode,
    data_format,
    *,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel, stride, padding = dtype_x_k_s

    if data_format == "NCW":
        x[0] = x[0].reshape((x[0].shape[0], x[0].shape[2], x[0].shape[1]))

    if padding == "VALID":
        ceil_mode = False

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        # data_format=data_format,
    )
