# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.nn.functional.binary_cross_entropy_with_logits",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=0,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=1.0013580322265625e-05,
        max_value=1.0,
        min_num_dims=1,
        max_num_dims=1,
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_binary_cross_entropy_with_logits(
    dtype_and_x,
    dtype_and_weight,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    # TODO: paddle's implementation of pos_weight is wrong
    # https://github.com/PaddlePaddle/Paddle/blob/f0422a28d75f9345fa3b801c01cd0284b3b44be3/python/paddle/nn/functional/loss.py#L831
    x_dtype, x = dtype_and_x
    weight_dtype, weight = dtype_and_weight
    helpers.test_frontend_function(
        input_dtypes=[
            x_dtype[0],
            weight_dtype[0],
        ],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        logit=x[0],
        label=x[0],
        weight=weight[0],
        reduction=reduction,
    )


# mse_loss
@handle_frontend_test(
    fn_tree="paddle.nn.functional.mse_loss",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=0.0,
        max_value=1.0,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=0.0,
        max_value=1.0,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_mse_loss(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true
    helpers.test_frontend_function(
        input_dtypes=[true_dtype[0], true_dtype[0]],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred[0],
        label=true[0],
        reduction=reduction,
    )
