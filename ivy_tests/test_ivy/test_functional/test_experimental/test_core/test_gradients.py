# global
from hypothesis import strategies as st
import pytest
import numpy as np

# local
from ivy_tests.test_ivy.test_functional.test_core.test_gradients import (
    get_gradient_arguments_with_lr,
)
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test, update_backend


# bind_custom_gradient_function
@pytest.mark.parametrize(
    "x_", [[[4.6, 2.1, 5], [2.8, 1.3, 6.2]], [[4.6, 2.1], [5, 2.8], [1.3, 6.2]]]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("inter_func_str", ["square", "cos"])
@pytest.mark.parametrize(
    "custom_grad_fn",
    [lambda *args: args[1] * args[0][0], lambda *args: args[1] * args[0][1]],
)
def test_bind_custom_gradient_function(
    x_, dtype, inter_func_str, custom_grad_fn, backend_fw
):
    if backend_fw == "numpy":
        return
    with update_backend(backend_fw) as ivy_backend:
        inter_func_ = lambda x: ivy_backend.__dict__[inter_func_str](x)
        x = ivy_backend.array(x_, dtype=dtype)
        inter_func = ivy_backend.bind_custom_gradient_function(
            inter_func_, custom_grad_fn
        )
        func = lambda x: ivy_backend.mean(ivy_backend.exp(inter_func(x)))
        ret, grad = ivy_backend.execute_with_gradients(func, x)
        ret_np = helpers.flatten_and_to_np(backend=backend_fw, ret=ret)
        grad_np = helpers.flatten_and_to_np(backend=backend_fw, ret=grad)

    with update_backend("tensorflow") as gt_backend:
        x = gt_backend.array(x_, dtype=dtype)
        inter_func_ = lambda x: gt_backend.__dict__[inter_func_str](x)
        inter_func = gt_backend.bind_custom_gradient_function(
            inter_func_, custom_grad_fn
        )
        func = lambda x: gt_backend.mean(gt_backend.exp(inter_func(x)))
        ret_gt, grad_gt = gt_backend.execute_with_gradients(func, x)
        ret_np_from_gt = helpers.flatten_and_to_np(backend="tensorflow", ret=ret_gt)
        grad_np_from_gt = helpers.flatten_and_to_np(backend="tensorflow", ret=grad_gt)

    for ret, ret_from_gt in zip(ret_np, ret_np_from_gt):
        assert np.allclose(ret, ret_from_gt)
    for grad, grad_from_gt in zip(grad_np, grad_np_from_gt):
        assert grad.shape == grad_from_gt.shape
        assert np.allclose(grad, grad_from_gt)


@handle_test(
    fn_tree="functional.ivy.experimental.adagrad_step",
    dtype_n_dcdw_n_vt=get_gradient_arguments_with_lr(
        num_arrays=2,
        no_lr=True,
        min_value=1e-05,
        max_value=1e08,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    epsilon=st.floats(min_value=1e-1, max_value=1),
)
def test_adagrad_step(
    *,
    dtype_n_dcdw_n_vt,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtypes, [dcdw, vt] = dtype_n_dcdw_n_vt
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        dcdw=dcdw,
        vt=vt,
        epsilon=epsilon,
    )


# adagrad_update
@handle_test(
    fn_tree="functional.ivy.experimental.adagrad_update",
    dtype_n_ws_n_dcdw_n_vt_n_lr=get_gradient_arguments_with_lr(
        num_arrays=3,
        min_value=1e-05,
        max_value=1e08,
        large_abs_safety_factor=2.0,
        small_abs_safety_factor=2.0,
    ),
    step=st.integers(min_value=1, max_value=10),
    epsilon=st.floats(min_value=1e-2, max_value=1),
    stopgrad=st.booleans(),
)
def test_adagrad_update(
    *,
    dtype_n_ws_n_dcdw_n_vt_n_lr,
    step,
    epsilon,
    stopgrad,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtypes, [w, dcdw, vt], lr = dtype_n_ws_n_dcdw_n_vt_n_lr
    stop_gradients = stopgrad
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        w=w,
        dcdw=dcdw,
        lr=lr,
        vt_tm1=vt,
        step=step,
        epsilon=epsilon,
        stop_gradients=stop_gradients,
    )
