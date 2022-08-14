# global
import pytest
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# cross_entropy
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
    epsilon=st.floats(min_value=0, max_value=0.49),
    num_positional_args=helpers.num_positional_args(fn_name="cross_entropy"),
    data=st.data(),
)
@handle_cmd_line_args
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


@pytest.mark.parametrize(
    "t_n_p_n_res", [([[0.0, 1.0, 0.0]], [[0.3, 0.2, 0.5]], [1.609438])]
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_cross_entropy_ground_truth(t_n_p_n_res, dtype, tensor_fn, device, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype=dtype, device=device)
    true = tensor_fn(true, dtype=dtype, device=device)
    ret = ivy.cross_entropy(true, pred)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert list(ret.shape) == [1]
    # value test
    assert np.allclose(call(ivy.cross_entropy, true, pred), np.asarray(true_target))


# binary_cross_entropy
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
    epsilon=st.floats(min_value=0, max_value=0.49),
    num_positional_args=helpers.num_positional_args(fn_name="binary_cross_entropy"),
    data=st.data(),
)
@handle_cmd_line_args
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


@pytest.mark.parametrize(
    "t_n_p_n_res",
    [([[0.0, 1.0, 0.0]], [[0.3, 0.7, 0.5]], [[0.35667494, 0.35667494, 0.69314718]])],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_binary_cross_entropy_ground_truth(t_n_p_n_res, dtype, tensor_fn, device, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype=dtype, device=device)
    true = tensor_fn(true, dtype=dtype, device=device)
    ret = ivy.binary_cross_entropy(true, pred)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == pred.shape
    # value test
    assert np.allclose(
        call(ivy.binary_cross_entropy, true, pred), np.asarray(true_target)
    )


# sparse_cross_entropy
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
    epsilon=st.floats(min_value=0, max_value=0.49),
    num_positional_args=helpers.num_positional_args(fn_name="sparse_cross_entropy"),
    data=st.data(),
)
@handle_cmd_line_args
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


@pytest.mark.parametrize("t_n_p_n_res", [([1], [[0.3, 0.2, 0.5]], [1.609438])])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_sparse_cross_entropy_ground_truth(t_n_p_n_res, dtype, tensor_fn, device, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype=dtype, device=device)
    true = ivy.array(true, dtype="int32", device=device)
    ret = ivy.sparse_cross_entropy(true, pred)
    # type test
    assert ivy.is_ivy_array(ret)
