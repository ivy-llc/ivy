# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# cross_entropy
@handle_test(
    fn_tree="functional.ivy.cross_entropy",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
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
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    axis=helpers.ints(min_value=-1, max_value=0),
    epsilon=helpers.floats(min_value=0, max_value=0.49),
)
def test_cross_entropy(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    axis,
    epsilon,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
    ground_truth_backend,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        test_gradients=test_gradients,
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
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1.0013580322265625e-05,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    epsilon=helpers.floats(min_value=0, max_value=0.49),
)
def test_binary_cross_entropy(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    epsilon,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
    ground_truth_backend,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        true=true[0],
        pred=pred[0],
        epsilon=epsilon,
        reduction=reduction,
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
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
    ground_truth_backend,
):
    true_dtype, true = dtype_and_true
    pred_dtype, pred = dtype_and_pred
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        test_gradients=test_gradients,
        true=true[0],
        pred=pred[0],
        axis=axis,
        epsilon=epsilon,
        reduction=reduction,
    )
