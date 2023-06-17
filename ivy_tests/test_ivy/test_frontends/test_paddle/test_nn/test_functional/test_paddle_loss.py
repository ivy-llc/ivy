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
    dtype_and_pos_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=0,
        max_value=10,
        exclude_min=True,
        exclude_max=True,
    ),
)
def test_paddle_binary_cross_entropy_with_logits(
    dtype_and_x,
    dtype_and_weight,
    reduction,
    dtype_and_pos_weight,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtype, x = dtype_and_x
    weight_dtype, weight = dtype_and_weight
    pos_weight_dtype, pos_weight = dtype_and_pos_weight
    helpers.test_frontend_function(
        input_dtypes=[
            x_dtype[0],
            weight_dtype[0],
            pos_weight_dtype[0],
        ],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        logit=x[0],
        label=x[0],
        weight=weight[0],
        reduction=reduction,
        pos_weight=pos_weight[0],
    )
