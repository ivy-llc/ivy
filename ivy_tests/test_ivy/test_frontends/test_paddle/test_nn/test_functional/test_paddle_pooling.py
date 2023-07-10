# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# max_pool1d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.max_pool1d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=3,
        explicit_or_str_padding=False,
        only_explicit_padding=True,
    ),
    test_with_out=st.just(False),
)
def test_paddle_max_pool1d(
    dtype_x_k_s,
    *,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, padding = dtype_x_k_s
    # Paddle ground truth func expects input to be consistent
    # with a channels first format i.e. NCW
    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], x[0].shape[1]))
    x_shape = [x[0].shape[2]]

    # Paddle ground truth func also takes padding input as an integer
    # or a tuple of integers, not a string
    padding = tuple(
        [
            ivy.functional.layers._handle_padding(
                x_shape[i], stride[0], kernel_size[i], padding
            )
            for i in range(len(x_shape))
        ]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
