"""Collection of tests for losses."""

# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method


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


@handle_method(
    method_tree="stateful.losses.DINOLoss.__call__",
    dtype_and_student_output=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        shape=(5,),
    ),
    dtype_and_teacher_output=helpers.dtype_and_values(
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
    student_temp=helpers.floats(min_value=0, max_value=1.0),
    center_momentum=helpers.floats(min_value=0, max_value=1.0),
    n_crops=helpers.ints(min_value=1, max_value=5),
    teacher_temp=helpers.floats(min_value=0, max_value=1.0),
    warmup_teacher_temp=helpers.floats(min_value=0, max_value=1.0),
    warmup_teacher_temp_epochs=helpers.ints(min_value=1, max_value=10),
    nepochs=helpers.ints(min_value=20, max_value=80),
    method_num_positional_args=helpers.num_positional_args(fn_name="DINOLoss._forward"),
)
def test_dino_loss(
    *,
    dtype_and_student_output,
    dtype_and_teacher_output,
    student_temp,
    center_momentum,
    n_crops,
    teacher_temp,
    warmup_teacher_temp,
    warmup_teacher_temp_epochs,
    nepochs,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
    on_device,
):
    out_dim = 1
    epochs = 5
    dtype_true, student_output = dtype_and_student_output
    dtype_pred, teacher_output = dtype_and_teacher_output
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        method_input_dtypes=dtype_true + dtype_pred,
        init_with_v={
            "out_dim": out_dim,
            "student_temp": student_temp,
            "center_momentum": center_momentum,
            "n_crops": n_crops,
            "teacher_temp": teacher_temp,
            "warmup_teacher_temp": warmup_teacher_temp,
            "warmup_teacher_temp_epochs": warmup_teacher_temp_epochs,
            "nepochs": nepochs,
        },
        method_with_v={
            "student_output": student_output[0],
            "teacher_output": teacher_output[0],
            "epochs": epochs,
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


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
