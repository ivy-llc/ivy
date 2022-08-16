# global
import numpy as np
from hypothesis import given, settings

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# cross_entropy
@handle_cmd_line_args
@given(
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_int_dtypes,
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=1.0013580322265625e-05,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    epsilon=helpers.floats(min_value=0, max_value=0.49),
    num_positional_args=helpers.num_positional_args(fn_name="cross_entropy"),
)
@settings(max_examples=1)
def test_cross_entropy(
    *,
    dtype_and_true,
    dtype_and_pred,
    axis,
    epsilon,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true

    helpers.test_function(
        input_dtypes=[true_dtype, pred_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cross_entropy",
        rtol_=1e-03,
        true=np.asarray(true, dtype=true_dtype),
        pred=np.asarray(pred, dtype=pred_dtype),
        axis=axis,
        epsilon=epsilon,
    )


# binary_cross_entropy
@handle_cmd_line_args
@given(
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_int_dtypes,
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=1.0013580322265625e-05,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    epsilon=helpers.floats(min_value=0, max_value=0.49),
    num_positional_args=helpers.num_positional_args(fn_name="binary_cross_entropy"),
)
@settings(max_examples=1)
def test_binary_cross_entropy(
    *,
    dtype_and_true,
    dtype_and_pred,
    epsilon,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true

    helpers.test_function(
        input_dtypes=[true_dtype, pred_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="binary_cross_entropy",
        true=np.asarray(true, dtype=true_dtype),
        pred=np.asarray(pred, dtype=pred_dtype),
        epsilon=epsilon,
    )


# sparse_cross_entropy
@handle_cmd_line_args
@given(
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_int_dtypes,
        min_value=0,
        max_value=2,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=1.0013580322265625e-05,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    epsilon=helpers.floats(min_value=0, max_value=0.49),
    num_positional_args=helpers.num_positional_args(fn_name="sparse_cross_entropy"),
)
@settings(max_examples=1)
def test_sparse_cross_entropy(
    *,
    dtype_and_true,
    dtype_and_pred,
    axis,
    epsilon,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    true_dtype, true = dtype_and_true
    pred_dtype, pred = dtype_and_pred

    helpers.test_function(
        input_dtypes=[true_dtype, pred_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sparse_cross_entropy",
        true=np.asarray(true, dtype=true_dtype),
        pred=np.asarray(pred, dtype=pred_dtype),
        axis=axis,
        epsilon=epsilon,
    )
