# global
from hypothesis import strategies as st

# local	# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.ivy.layers import _handle_padding


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
)
def test_paddle_avg_pool1d(
    dtype_x_k_s,
    ceil_mode,
    *,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel, stride, padding = dtype_x_k_s

    if len(stride) == 1:
        stride = stride[0]

    if isinstance(padding, str):
        if padding == "SAME":
            padding = [
                _handle_padding(x[0].shape[i + 1], stride[i], kernel[i], padding)
                for i in range(1)
            ]
        else:
            padding = 0
    elif len(padding) == 1:
        pass
    else:
        padding = 0

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
    )
