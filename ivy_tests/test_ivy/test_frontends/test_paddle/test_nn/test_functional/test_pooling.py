# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_nn.test_functional import (
    test_pooling_functions,
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
    divisor_override=st.one_of(st.none(), st.integers(min_value=1, max_value=4)),
)
def test_paddle_avg_pool2d(
    dtype_x_k_s,
    exclusive,
    ceil_mode,
    data_format,
    divisor_override,
    *,
    test_flags,
    backend_fw,
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
        padding = test_pooling_functions.calculate_same_padding(
            kernel, stride, x[0].shape[2:]
        )
    else:
        padding = (0, 0)

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        exclusive=exclusive,
        divisor_override=divisor_override,
        data_format=data_format,
    )


# avg_pool1d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.avg_pool1d",
    x_k_s_p_df=helpers.arrays_for_pooling(
        min_dims=3,
        max_dims=3,
        min_side=1,
        max_side=4,
    ),
    exclusive=st.booleans(),
    ceil_mode=st.just(False),
    test_with_out=st.just(False),
)
def test_paddle_avg_pool1d(
    *,
    x_k_s_p_df,
    frontend,
    test_flags,
    backend_fw,
    on_device,
    fn_tree,
    exclusive,
    ceil_mode,
):
    (input_dtype, x, kernel_size, stride, padding) = x_k_s_p_df
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        exclusive=exclusive,
        ceil_mode=ceil_mode,
    )


# adaptive_avg_pool1d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.adaptive_avg_pool1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=1,
        max_value=100,
        min_value=-100,
    ),
    output_size=helpers.ints(min_value=1, max_value=5),
)
def test_paddle_adaptive_avg_pool1d(
    *,
    dtype_and_x,
    output_size,
    test_flags,
    frontend,
    on_device,
    backend_fw,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        output_size=output_size,
    )


# adaptive_avg_pool2d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.adaptive_avg_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
        max_value=100,
        min_value=-100,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
        ),
        helpers.ints(min_value=1, max_value=5),
    ),
)
def test_paddle_adaptive_avg_pool2d(
    *,
    dtype_and_x,
    output_size,
    test_flags,
    frontend,
    on_device,
    backend_fw,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        output_size=output_size,
    )


# adaptive_max_pool2d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.adaptive_max_pool2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
        # Setting max and min value because this operation in paddle is not
        # numerically stable
        max_value=100,
        min_value=-100,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
        ),
        helpers.ints(min_value=1, max_value=5),
    ),
)
def test_paddle_adaptive_max_pool2d(
    *,
    dtype_and_x,
    output_size,
    test_flags,
    frontend,
    on_device,
    backend_fw,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        output_size=output_size,
    )


# adaptive_max_pool3d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.adaptive_max_pool3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=5,
        max_num_dims=5,
        min_dim_size=1,
        # Setting max and min value because this operation in paddle is not
        # numerically stable
        max_value=100,
        min_value=-100,
    ),
    output_size=st.one_of(
        st.tuples(
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
            helpers.ints(min_value=1, max_value=5),
        ),
        helpers.ints(min_value=1, max_value=5),
    ),
)
def test_paddle_adaptive_max_pool3d(
    *,
    dtype_and_x,
    output_size,
    test_flags,
    frontend,
    on_device,
    backend_fw,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        output_size=output_size,
    )

# max_unpool1d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.max_unpool1d",
    x_k_s_p=helpers.arrays_for_pooling(min_dims=3, max_dims=3, min_side=1, max_side=4),
    indices=st.lists(st.integers(0, 1), min_size=1, max_size=4),
)
def test_paddle_max_unpool1d(
    *,
    x_k_s_p,
    indices,
    test_flags,
    frontend,
    on_device,
    backend_fw,
    fn_tree,
):
    (input_dtype, x, kernel_size, stride, padding) = x_k_s_p
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        indices=indices,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    