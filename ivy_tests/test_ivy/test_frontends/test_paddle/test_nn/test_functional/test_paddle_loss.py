# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# binary_cross_entropy
@handle_frontend_test(
    fn_tree="paddle.nn.loss.binary_cross_entropy",
    dtype_and_vals=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        shared_dtype=True,
        min_value=1.0013580322265625e-05,
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
        shape=(5,),
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_binary_cross_entropy(
    *,
    dtype_and_vals,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_vals
    pred_dtype, pred = input_dtype[0], x[0]
    true_dtype, true = input_dtype[1], x[1]
    weight_dtype, weight = input_dtype[2], x[2]

    helpers.test_frontend_function(
        input_dtypes=[pred_dtype, true_dtype, weight_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred,
        label=true,
        weight=weight,
        reduction=reduction,
        rtol=1e-02,
        atol=1e-02,
    )
