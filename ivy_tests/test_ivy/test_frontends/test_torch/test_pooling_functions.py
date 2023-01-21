# global
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_frontend_test


# avg_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.avg_pool2d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=4,
        max_dims=4,
        min_side=1,
        max_side=4,
    ),
)
def test_torch_avg_pool2d(
    dtype_x_k_s,
    *,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, padding = dtype_x_k_s

    # Torch ground truth func expects input to be consistent
    # with a channels first format i.e. NCHW
    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], *x[0].shape[1:-1]))
    x_shape = list(x[0].shape[2:])

    # Torch ground truth func also takes padding input as an integer
    # or a tuple of integers, not a string
    padding = tuple(
        [
            ivy.handle_padding(x_shape[i], stride[0], kernel_size[i], padding)
            for i in range(len(x_shape))
        ]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )
