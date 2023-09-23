# global
import pytest
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.pipeline_helper import BackendHandler


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
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        inter_func_ = lambda x: ivy_backend.__dict__[inter_func_str](x)
        x = ivy_backend.array(x_, dtype=dtype)
        inter_func = ivy_backend.bind_custom_gradient_function(
            inter_func_, custom_grad_fn
        )
        func = lambda x: ivy_backend.mean(ivy_backend.exp(inter_func(x)))
        ret, grad = ivy_backend.execute_with_gradients(func, x)
        ret_np = helpers.flatten_and_to_np(backend=backend_fw, ret=ret)
        grad_np = helpers.flatten_and_to_np(backend=backend_fw, ret=grad)

    with BackendHandler.update_backend("tensorflow") as gt_backend:
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
