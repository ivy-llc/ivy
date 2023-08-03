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


# Binary Cross Entropy Loss
@handle_method(
    method_tree="stateful.losses.BinaryCrossEntropyLoss.__call__",
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
    method_num_positional_args=helpers.num_positional_args(
        fn_name="BinaryCrossEntropyLoss._forward"
    ),
)
def test_binary_cross_entropy_loss(
    *,
    dtype_and_true,
    dtype_and_pred,
    dtype_and_pos,
    from_logits,
    reduction,
    axis,
    epsilon,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    dtype_true, true = dtype_and_true
    dtype_pred, pred = dtype_and_pred
    dtype_pos_weight, pos_weight = dtype_and_pos

    if from_logits:
        helpers.test_method(
            ground_truth_backend=ground_truth_backend,
            init_flags=init_flags,
            method_flags=method_flags,
            method_input_dtypes=dtype_true + dtype_pred + dtype_pos_weight,
            init_all_as_kwargs_np={
                "from_logits": from_logits,
                "epsilon": epsilon,
                "reduction": reduction,
            },
            method_all_as_kwargs_np={
                "true": true[0],
                "pred": pred[0],
                "pos_weight": pos_weight[0],
                "axis": axis,
            },
            class_name=class_name,
            method_name=method_name,
            rtol_=1e-2,
            atol_=1e-2,
            on_device=on_device,
        )
    else:
        helpers.test_method(
            ground_truth_backend=ground_truth_backend,
            init_flags=init_flags,
            method_flags=method_flags,
            method_input_dtypes=dtype_true + dtype_pred,
            init_all_as_kwargs_np={
                "from_logits": from_logits,
                "epsilon": epsilon,
                "reduction": reduction,
            },
            method_all_as_kwargs_np={
                "true": true[0],
                "pred": pred[0],
                "axis": axis,
            },
            class_name=class_name,
            method_name=method_name,
            rtol_=1e-2,
            atol_=1e-2,
            on_device=on_device,
        )


# L1 Loss (Mean Absolute Error)
@handle_method(
    method_tree="stateful.losses.L1Loss.__call__",
    dtype_and_targets=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=3,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_predictions=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=3,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    reduction=st.sampled_from([ "mean", "sum"]),
)
def test_l1_loss(
    *,
    dtype_and_targets,
    dtype_and_predictions,
    reduction,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    targets_dtype, targets = dtype_and_targets
    predictions_dtype, predictions = dtype_and_predictions
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        method_input_dtypes=targets_dtype + predictions_dtype,
        init_all_as_kwargs_np={
            "reduction": reduction,
        },
        method_all_as_kwargs_np={"targets": targets[0], "predictions": predictions[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )
