# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_pooling_functions import (
    calculate_same_padding,
)


# avg_pool2d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.pooling.avg_pool2d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=4,
        max_dims=4,
        min_side=1,
        max_side=4,
    ),
    ceil_mode=st.booleans(),
    exclusive=st.booleans(),
    data_format=st.sampled_from(["NCHW", "NHWC"]),
)
def test_paddle_avg_pool2d(
    dtype_x_k_s,
    exclusive,
    ceil_mode,
    data_format,
    *,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel, stride, padding = dtype_x_k_s

    if data_format == "NCHW":
        x[0] = x[0].reshape(
            (x[0].shape[0], x[0].shape[3], x[0].shape[1], x[0].shape[2])
        )

    if len(stride) == 1:
        stride = (stride[0], stride[0])

    if padding == "SAME":
        padding = calculate_same_padding(kernel, stride, x[0].shape[2:])
    else:
        padding = (0, 0)

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        exclusive=exclusive,
        divisor_override=None,
        data_format=data_format,
    )
