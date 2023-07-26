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
    compute_full_loss=st.sampled_from([True, False]),
)
def test_log_poisson_loss(
    dtype_and_targets,
    dtype_and_log_input,
    compute_full_loss,
    test_gradients,
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
        init_input_dtypes=targets_dtype,
        method_input_dtypes=log_input_dtype,
        init_all_as_kwargs_np={
            "compute_full_loss": compute_full_loss,
            "axis": -1,
            "reduction": "none",
        },
        method_all_as_kwargs_np={"true": targets, "pred": log_input},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        on_device=on_device,
    )
