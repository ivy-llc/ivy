"""Collection of tests for losses."""

# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method


# Log Poisson Loss
@handle_method(
    method_tree="stateful.losses.LogPoissonLoss.__call__",
    dtype_and_targets=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=3,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_log_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        min_value=0,
        max_value=3,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    axis=st.integers(min_value=-1, max_value=1),
    compute_full_loss=st.sampled_from([True, False]),
    method_num_positional_args=helpers.num_positional_args(
        fn_name="LogPoissonLoss._forward"
    ),
    reduction=st.sampled_from(["none", "mean", "sum"]),
)
def test_log_poisson_loss(
    *,
    dtype_and_targets,
    dtype_and_log_input,
    compute_full_loss,
    axis,
    reduction,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    targets_dtype, targets = dtype_and_targets
    log_input_dtype, log_input = dtype_and_log_input
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        method_input_dtypes=targets_dtype + log_input_dtype,
        init_all_as_kwargs_np={
            "compute_full_loss": compute_full_loss,
            "axis": axis,
            "reduction": reduction,
        },
        method_all_as_kwargs_np={"true": targets[0], "pred": log_input[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


# Cross Entropy Loss
@handle_method(
    method_tree="stateful.losses.CrossEntropyLoss.__call__",
    dtype_and_targets=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=3,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_log_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        min_value=0,
        max_value=3,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    axis=st.integers(min_value=-1, max_value=1),
    method_num_positional_args=helpers.num_positional_args(
        fn_name="CrossEntropyLoss._forward"
    ),
    reduction=st.sampled_from(["none", "mean", "sum"]),
)
def test_cross_entropy_loss(
    *,
    dtype_and_targets,
    dtype_and_log_input,
    axis,
    reduction,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    targets_dtype, targets = dtype_and_targets
    log_input_dtype, log_input = dtype_and_log_input
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        method_input_dtypes=targets_dtype + log_input_dtype,
        init_all_as_kwargs_np={
            "axis": axis,
            "reduction": reduction,
        },
        method_all_as_kwargs_np={"true": targets[0], "pred": log_input[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


# Dice Loss
@handle_test(
    fn_tree="functional.ivy.dice_loss",
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    smooth=helpers.floats(min_value=0, max_value=1.0),
    axis=st.integers(min_value=-1, max_value=1),
)
def test_dice_loss(
    dtype_and_pred,
    dtype_and_target,
    reduction,
    smooth,
    axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    pred_dtype, pred = dtype_and_pred
    target_dtype, target = dtype_and_target

    helpers.test_function(
        input_dtypes=pred_dtype + target_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        pred=pred[0],
        target=target[0],
        reduction=reduction,
        smooth=smooth,
        axis=axis,
    )

