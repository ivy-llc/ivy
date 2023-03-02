# global
from hypothesis import strategies as st

# local
import ivy
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
    test_with_out=st.just(False),
)
def test_torch_avg_pool2d(
    dtype_x_k_s,
    *,
    test_flags,
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
        test_flags=test_flags,
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


# max_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.max_pool2d",
    x_k_s_p=helpers.arrays_for_pooling(
        min_dims=4,
        max_dims=4,
        min_side=1,
        max_side=4,
        allow_explicit_padding=True,
        return_dilation=True,
    ).filter(lambda x: x[4] != "VALID" and x[4] != "SAME"),
    test_with_out=st.just(False),
    ceil_mode=st.just(True),
)
def test_torch_max_pool2d(
    x_k_s_p,
    ceil_mode,
    *,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    dtype, x, kernel, stride, pad, dilation = x_k_s_p
    # Torch ground truth func expects input to be consistent
    # with a channels first format i.e. NCHW
    x[0] = x[0].reshape((x[0].shape[0], x[0].shape[-1], *x[0].shape[1:-1]))
    pad = (pad[0][0], pad[1][0])

    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel,
        stride=stride,
        padding=pad,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


# adaptive_avg_pool1d
@handle_frontend_test(
    fn_tree="torch.nn.functional.adaptive_avg_pool1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=5,
        max_value=100,
        min_value=-100,
    ),
    output_size=helpers.ints(min_value=1, max_value=10),
    test_with_out=st.just(False),
)
def test_torch_adaptive_avg_pool1d(
    *,
    dtype_and_x,
    output_size,
    on_device,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        output_size=output_size,
        atol=1e-2,
    )


# adaptive_avg_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.adaptive_avg_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=4,
        min_dim_size=5,
        max_value=100,
        min_value=-100,
    ),
    output_size=st.tuples(
        helpers.ints(min_value=1, max_value=10),
        helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_torch_adaptive_avg_pool2d(
    *,
    dtype_and_x,
    output_size,
    on_device,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        output_size=output_size,
        atol=1e-2,
    )
