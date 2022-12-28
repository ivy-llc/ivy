# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# binary_cross_entropy_with_logits
@handle_test(
    fn_tree="functional.ivy.experimental.binary_cross_entropy_with_logits",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_pos_weight=st.one_of(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_value=0,
            max_value=1,
            allow_inf=False,
            exclude_min=True,
            exclude_max=True,
            min_num_dims=1,
            max_num_dims=1,
            min_dim_size=2,
        ),
        st.just([[None], [None]]),
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    epsilon=helpers.floats(min_value=0, max_value=0.49),
)
def test_binary_cross_entropy_with_logits(
    dtype_and_true,
    dtype_and_pred,
    dtype_and_pos_weight,
    reduction,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true
    pos_weight_dtype, pos_weight = dtype_and_pos_weight
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        true=true[0],
        pred=pred[0],
        epsilon=epsilon,
        pos_weight=pos_weight[0],
        reduction=reduction,
    )
