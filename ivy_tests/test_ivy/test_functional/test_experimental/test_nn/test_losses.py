# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test

    
#mse
@handle_test(
    fn_tree="functional.ivy.experimental.mse_loss",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
)
def test_mse_loss(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true

    # Unpack lists if necessary
    if isinstance(true, list):
        true = true[0]
    if isinstance(pred, list):
        pred = pred[0]
        
    # Adjust shapes for broadcasting
    while len(pred.shape) < len(true.shape):
        pred = pred.unsqueeze(0)
    while len(pred.shape) > len(true.shape):
        true = true.unsqueeze(0)

    # Add dimension handling
    if len(pred.shape) > len(true.shape):
        pred = pred[: true.ndim]
    elif len(pred.shape) < len(true.shape):
        true = true[: pred.ndim]
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        true=true,
        pred=pred,
        reduction=reduction,
    )

#mae
@handle_test(
    fn_tree="functional.ivy.experimental.mae_loss",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
)
def test_mae_loss(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true

    # Unpack lists if necessary
    if isinstance(true, list):
        true = true[0]
    if isinstance(pred, list):
        pred = pred[0]
    # Adjust shapes for broadcasting
    while len(pred.shape) < len(true.shape):
	pred = pred.unsqueeze(0)
    while len(pred.shape) > len(true.shape):
	true = true.unsqueeze(0)

    # Add dimension handling
    if len(pred.shape) > len(true.shape):
        pred = pred[: true.ndim]
    elif len(pred.shape) < len(true.shape):
        true = true[: pred.ndim]
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        true=true,
        pred=pred,
        reduction=reduction,
    )
