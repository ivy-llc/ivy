# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# log_poisson_loss
@handle_test(
    fn_tree="functional.ivy.experimental.log_poisson_loss",
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
    test_with_out=st.just(False),
)
def test_log_poisson_loss(
    *,
    dtype_and_targets,
    dtype_and_log_input,
    compute_full_loss,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    targets_dtype, targets = dtype_and_targets
    log_input_dtype, log_input = dtype_and_log_input
    helpers.test_function(
        input_dtypes=targets_dtype + log_input_dtype,
        test_flags=test_flags,
        backend_to_fix=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        targets=targets[0],
        log_input=log_input[0],
        compute_full_loss=compute_full_loss,
        atol_=1e-2,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.l1_loss",
    dtype_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
        max_value=100,
        allow_inf=False,
    ),
    dtype_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
        max_value=100,
        allow_inf=False,
    ),
    reduction=st.sampled_from(["sum", "mean", "none"]),
)
def test_l1_loss(
    *,
    dtype_input,
    dtype_target,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype_input, input = dtype_input
    dtype_target, target = dtype_target

    helpers.test_function(
        input_dtypes=dtype_input + dtype_target,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-02,
        input=input[0],
        target=target[0],
        reduction=reduction,
    )


# smooth_l1_loss
# all loss functions failing for paddle backend due to
# "There is no grad op for inputs:[0] or it's stop_gradient=True."
@handle_test(
    fn_tree="functional.ivy.experimental.smooth_l1_loss",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10.0,
        max_value=10.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10.0,
        max_value=10.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    beta=helpers.floats(min_value=0.0, max_value=1.0),
    reduction=st.sampled_from(["none", "sum", "mean"]),
)
def test_smooth_l1_loss(
    dtype_and_input,
    dtype_and_target,
    beta,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype_input, input = dtype_and_input
    dtype_target, target = dtype_and_target

    helpers.test_function(
        input_dtypes=dtype_input + dtype_target,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        input=input[0],
        target=target[0],
        beta=beta,
        reduction=reduction,
    )
