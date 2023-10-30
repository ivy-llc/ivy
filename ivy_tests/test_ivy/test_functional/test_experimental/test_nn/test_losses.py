# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.binary_cross_entropy",
    dtype_input_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
        shape=(5,),
    ),
    reduction=st.sampled_from(["sum", "mean", "none"]),
    test_with_out=st.just(False),
    test_gradients=st.just(
        False
    ),  # value_test are failing if this is set to `True` # noqa
    ground_truth_backend="torch",
)
def test_binary_cross_entropy(
    dtype_input_target,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype_input, inputs = dtype_input_target

    helpers.test_function(
        input_dtypes=dtype_input,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-02,
        rtol_=1e-05,
        input=inputs[0],
        target=inputs[1],
        reduction=reduction,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.cross_entropy",
    dtype_input_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
        shape=(5,),
    ),
    reduction=st.sampled_from(["sum", "mean", "none"]),
    test_with_out=st.just(False),
    test_gradients=st.just(
        False
    ),  # value_test are failing if this is set to `True` # noqa
    ground_truth_backend="torch",
)
def test_cross_entropy_loss(
    dtype_input_target,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, inputs = dtype_input_target
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fn_name=fn_name,
        backend_to_test=backend_fw,
        on_device=on_device,
        atol_=1e-02,
        rtol_=1e-05,
        input=inputs[0],
        target=inputs[1],
        reduction=reduction,
    )


# huber_loss
@handle_test(
    fn_tree="functional.ivy.experimental.huber_loss",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    delta=helpers.floats(min_value=0.01, max_value=2.0),
)
def test_huber_loss(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    delta,
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
        reduction=reduction,
        delta=delta,
    )


# kl_div
@handle_test(
    fn_tree="functional.ivy.experimental.kl_div",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    reduction=st.sampled_from(["none", "sum", "batchmean", "mean"]),
    log_target=st.booleans(),
)
def test_kl_div(
    dtype_and_input,
    dtype_and_target,
    reduction,
    log_target,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, input = dtype_and_input
    input[0] = np.log(input[0])
    target_dtype, target = dtype_and_target
    if log_target:
        target[0] = np.log(target[0])
    helpers.test_function(
        input_dtypes=input_dtype + target_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-02,
        input=input[0],
        target=target[0],
        reduction=reduction,
        log_target=log_target,
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
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        targets=targets[0],
        log_input=log_input[0],
        compute_full_loss=compute_full_loss,
        atol_=1e-2,
    )


# poisson_nll_loss
@handle_test(
    fn_tree="functional.ivy.experimental.poisson_nll_loss",
    dtype_input_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_dim_size=1,
        min_num_dims=1,
        min_value=0,
        max_value=100,
        num_arrays=2,
        shared_dtype=True,
    ),
    log_input=st.booleans(),
    full=st.booleans(),
    epsilon=st.sampled_from([1e-8, 1e-5, 1e-3]),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    test_with_out=st.just(False),
    test_gradients=st.just(
        False
    ),  # value_test are failing if this is set to `True` # noqa
    ground_truth_backend="torch",
)
def test_poisson_nll_loss(
    dtype_input_target,
    log_input,
    full,
    epsilon,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, inputs = dtype_input_target
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        input=inputs[0],
        target=inputs[1],
        log_input=log_input,
        full=full,
        eps=epsilon,
        reduction=reduction,
        rtol_=1e-05,
        atol_=1e-05,
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
    ground_truth_backend="torch",
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


# soft_margin_loss
@handle_test(
    fn_tree="functional.ivy.experimental.soft_margin_loss",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
)
def test_soft_margin_loss(
    dtype_and_input,
    dtype_and_target,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, input = dtype_and_input
    target_dtype, target = dtype_and_target

    helpers.test_function(
        input_dtypes=input_dtype + target_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        pred=input[0],
        target=target[0],
        reduction=reduction,
    )
