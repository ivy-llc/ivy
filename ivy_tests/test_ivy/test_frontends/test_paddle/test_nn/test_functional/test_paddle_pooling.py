# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import math


def is_same_padding(padding, stride, kernel_size, input_shape):
    output_shape = tuple(
        [
            (input_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
            for i in range(len(padding))
        ]
    )
    return all(
        [
            output_shape[i] == math.ceil(input_shape[i] / stride[i])
            for i in range(len(padding))
        ]
    )


def calculate_same_padding(kernel_size, stride, shape):
    padding = tuple(
        [
            max(
                0,
                math.ceil(((shape[i] - 1) * stride[i] + kernel_size[i] - shape[i]) / 2),
            )
            for i in range(len(kernel_size))
        ]
    )
    if all([kernel_size[i] / 2 >= padding[i] for i in range(len(kernel_size))]):
        if is_same_padding(padding, stride, kernel_size, shape):
            return padding
    return 0, 0


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
)
def test_paddle_avg_pool2d(
    dtype_x_k_s,
    exclusive,
    ceil_mode,
    *,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel, stride, padding = dtype_x_k_s

    if len(stride) == 1:
        stride = (stride[0], stride[0])

    if padding == "SAME":
        padding = calculate_same_padding(kernel, stride, x[0].shape[2:])
    else:
        padding = (0, 0)

    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], *x[0].shape[1:-1]))

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
        exclusive=exclusive,
        divisor_override=None,
    )
