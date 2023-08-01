# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# cross_entropy
@handle_test(
    fn_tree="functional.ivy.cross_entropy",
    dtype_true_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        valid_axis=True,
        allow_neg_axes=True,
        force_int_axis=True,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    epsilon=helpers.floats(min_value=0.0, max_value=1.0),
)
def test_cross_entropy(
    dtype_true_axis,
    dtype_and_pred,
    reduction,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true, axis = dtype_true_axis

    helpers.test_function(
        input_dtypes=true_dtype + pred_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        true=true[0],
        pred=pred[0],
        axis=axis,
        epsilon=epsilon,
        reduction=reduction,
    )


# binary_cross_entropy
@handle_test(
    fn_tree="functional.ivy.binary_cross_entropy",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        shape=(5,),
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        shape=(5,),
    ),
    dtype_and_pos=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        shape=(5,),
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    axis=helpers.ints(min_value=-1, max_value=0),
    epsilon=helpers.floats(min_value=0, max_value=1.0),
    from_logits=st.booleans(),
)
def test_binary_cross_entropy(
    dtype_and_true,
    dtype_and_pred,
    dtype_and_pos,
    from_logits,
    reduction,
    axis,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype_true, true = dtype_and_true
    dtype_pred, pred = dtype_and_pred
    dtype_pos_weight, pos_weight = dtype_and_pos

    if from_logits:
        helpers.test_function(
            input_dtypes=dtype_true + dtype_pred + dtype_pos_weight,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            on_device=on_device,
            rtol_=1e-02,
            atol_=1e-02,
            true=true[0],
            pred=pred[0],
            axis=axis,
            epsilon=epsilon,
            reduction=reduction,
            from_logits=from_logits,
            pos_weight=pos_weight[0],
        )
    else:
        helpers.test_function(
            input_dtypes=dtype_true + dtype_pred,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            on_device=on_device,
            rtol_=1e-02,
            atol_=1e-02,
            true=true[0],
            pred=pred[0],
            axis=axis,
            epsilon=epsilon,
            reduction=reduction,
            from_logits=from_logits,
        )


# sparse_cross_entropy
@handle_test(
    fn_tree="functional.ivy.sparse_cross_entropy",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=0,
        max_value=2,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    axis=helpers.ints(min_value=-1, max_value=0),
    epsilon=helpers.floats(min_value=0.01, max_value=0.49),
)
def test_sparse_cross_entropy(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    axis,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    true_dtype, true = dtype_and_true
    pred_dtype, pred = dtype_and_pred
    helpers.test_function(
        input_dtypes=true_dtype + pred_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        true=true[0],
        pred=pred[0],
        axis=axis,
        epsilon=epsilon,
        reduction=reduction,
    )



