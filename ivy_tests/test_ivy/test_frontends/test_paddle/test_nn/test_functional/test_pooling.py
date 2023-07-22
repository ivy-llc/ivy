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
    exclusive=st.booleans(),
    ceil_mode=st.just(False),
    test_with_out=st.just(False),
)
def test_paddle_avg_pool1d(
    *,
    x_k_s_p_df,
    frontend,
    test_flags,
    on_device,
    fn_tree,
    exclusive,
    ceil_mode,
):
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
        exclusive=exclusive,
        ceil_mode=ceil_mode,
    )


# avg_pool3d
@handle_frontend_test(
    fn_tree="paddle.nn.functional.pooling.avg_pool3d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=5,
        max_dims=5,
        min_side=1,
        max_side=5,
    ),
    ceil_mode=st.booleans(),
    divisor_override=st.one_of(st.none(), st.integers(min_value=1, max_value=5)),
    data_format=st.sampled_from(["NCDHW", "NDHWC"]),
)
def test_paddle_avg_pool3d(
    dtype_x_k_s,
    ceil_mode,
    divisor_override,
    data_format,
    *,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel, stride, padding = dtype_x_k_s

    if len(stride) == 1:
        stride = (stride[0], stride[0], stride[0])

    if data_format == "NCDHW":
        x[0] = x[0].reshape(
            x[0].shape[0], x[0].shape[4], x[0].shape[1], x[0].shape[2], x[0].shape[3]
        )

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
        # divisor_override=divisor_override,
        data_format=data_format,
    )
