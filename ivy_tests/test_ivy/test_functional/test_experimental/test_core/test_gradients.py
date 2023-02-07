# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


# bind_custom_gradient_function
@pytest.mark.parametrize(
    "x_", [[[4.6, 2.1, 5], [2.8, 1.3, 6.2]], [[4.6, 2.1], [5, 2.8], [1.3, 6.2]]]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("inter_func_", [lambda x: ivy.square(x), lambda x: ivy.cos(x)])
@pytest.mark.parametrize(
    "custom_grad_fn",
    [lambda *args: args[1] * args[0][0], lambda *args: args[1] * args[0][1]],
)
def test_bind_custom_gradient_function(
    x_, dtype, inter_func_, custom_grad_fn, backend_fw
):
    fw = backend_fw.current_backend_str()
    if fw == "numpy":
        return
    x = ivy.array(x_, dtype=dtype)
    inter_func = ivy.bind_custom_gradient_function(inter_func_, custom_grad_fn)
    func = lambda x: ivy.mean(ivy.exp(inter_func(x)))
    ret, grad = ivy.execute_with_gradients(func, x)
    ret_np = helpers.flatten_and_to_np(ret=ret)
    grad_np = helpers.flatten_and_to_np(ret=grad)
    ivy.set_backend("tensorflow")
    x = ivy.array(x_, dtype=dtype)
    inter_func = ivy.bind_custom_gradient_function(inter_func_, custom_grad_fn)
    func = lambda x: ivy.mean(ivy.exp(inter_func(x)))
    ret_gt, grad_gt = ivy.execute_with_gradients(func, x)
    ret_np_from_gt = helpers.flatten_and_to_np(ret=ret_gt)
    grad_np_from_gt = helpers.flatten_and_to_np(ret=grad_gt)
    ivy.unset_backend()
    for ret, ret_from_gt in zip(ret_np, ret_np_from_gt):
        assert np.allclose(ret, ret_from_gt)
    for grad, grad_from_gt in zip(grad_np, grad_np_from_gt):
        assert grad.shape == grad_from_gt.shape
        assert np.allclose(grad, grad_from_gt)
