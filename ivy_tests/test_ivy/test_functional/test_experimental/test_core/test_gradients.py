# global
import pytest
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.pipeline_helper import BackendHandler


# --- Helpers --- #
# --------------- #


def _get_primals_and_tangents(x_, dtype, ivy_backend, primals_cont, tangents_cont):
    if primals_cont:
        primals = ivy_backend.Container(
            {
                "l": {
                    "a": ivy_backend.array(x_[0][0], dtype=dtype),
                    "b": ivy_backend.array(x_[0][1], dtype=dtype),
                }
            }
        )
    else:
        primals = ivy_backend.array(x_[0], dtype=dtype)

    if tangents_cont:
        tangents = ivy_backend.Container(
            {
                "l": {
                    "a": ivy_backend.array([t[0] for t in x_[1]], dtype=dtype),
                    "b": ivy_backend.array([t[0] for t in x_[1]], dtype=dtype),
                }
            }
        )
    else:
        if primals_cont:
            tangents = ivy_backend.array([t[0] for t in x_[1]], dtype=dtype)
        else:
            tangents = ivy_backend.array(x_[1], dtype=dtype).T
    return primals, tangents


# --- Main --- #
# ------------ #


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

        def inter_func_(x):
            return ivy_backend.__dict__[inter_func_str](x)

        x = ivy_backend.array(x_, dtype=dtype)
        inter_func = ivy_backend.bind_custom_gradient_function(
            inter_func_, custom_grad_fn
        )

        def func(x):
            return ivy_backend.mean(ivy_backend.exp(inter_func(x)))

        ret, grad = ivy_backend.execute_with_gradients(func, x)
        ret_np = helpers.flatten_and_to_np(backend=backend_fw, ret=ret)
        grad_np = helpers.flatten_and_to_np(backend=backend_fw, ret=grad)

    with BackendHandler.update_backend("tensorflow") as gt_backend:
        x = gt_backend.array(x_, dtype=dtype)

        def inter_func_(x):
            return gt_backend.__dict__[inter_func_str](x)

        inter_func = gt_backend.bind_custom_gradient_function(
            inter_func_, custom_grad_fn
        )

        def func(x):
            return gt_backend.mean(gt_backend.exp(inter_func(x)))

        ret_gt, grad_gt = gt_backend.execute_with_gradients(func, x)
        ret_np_from_gt = helpers.flatten_and_to_np(backend="tensorflow", ret=ret_gt)
        grad_np_from_gt = helpers.flatten_and_to_np(backend="tensorflow", ret=grad_gt)

    for ret, ret_from_gt in zip(ret_np, ret_np_from_gt):
        assert np.allclose(ret, ret_from_gt)
    for grad, grad_from_gt in zip(grad_np, grad_np_from_gt):
        assert grad.shape == grad_from_gt.shape
        assert np.allclose(grad, grad_from_gt)


# write a test for jvp
@pytest.mark.parametrize(
    "x_", [[[[4.6, 2.1, 5], [2.8, 1.3, 6.2]], [[4.6, 2.1], [5, 2.8], [1.3, 6.2]]]]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("func_str", ["square", "cos"])
@pytest.mark.parametrize(
    "nested_structs", ["nested_input_nested_output", "nested_input_flat_output", "none"]
)
def test_jvp(x_, dtype, func_str, backend_fw, nested_structs):
    if backend_fw in ["numpy", "paddle", "mxnet"]:
        pytest.skip()

    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        base_func = ivy_backend.__dict__[func_str]
        if nested_structs == "none":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, ivy_backend, False, False
            )
            func = base_func
        elif nested_structs == "nested_input_nested_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, ivy_backend, True, True
            )
            func = base_func
        elif nested_structs == "nested_input_flat_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, ivy_backend, True, True
            )

            def func(x):
                return base_func(x["l"]["a"]) + base_func(x["l"]["b"])

        primals = (primals,)
        tangents = (tangents,)
        primals_out, jvp = ivy_backend.jvp(func, primals, tangents)
        flat_primals_np = helpers.flatten_and_to_np(ret=primals_out, backend=backend_fw)
        jvp_np = helpers.flatten_and_to_np(ret=jvp, backend=backend_fw)
        assert jvp_np != []

    with BackendHandler.update_backend("jax") as gt_backend:
        base_func = gt_backend.__dict__[func_str]
        if nested_structs == "none":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, gt_backend, False, False
            )
            func = base_func
        elif nested_structs == "nested_input_nested_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, gt_backend, True, True
            )
            func = base_func
        elif nested_structs == "nested_input_flat_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, gt_backend, True, True
            )

            def func(x):
                return base_func(x["l"]["a"]) + base_func(x["l"]["b"])

        # func = base_func

        primals = (primals,)
        tangents = (tangents,)
        primals_out_gt, jvp = gt_backend.jvp(func, primals, tangents)
        flat_primals_np_from_gt = helpers.flatten_and_to_np(
            ret=primals_out_gt, backend="jax"
        )
        jvp_np_from_gt = helpers.flatten_and_to_np(ret=jvp, backend="jax")
        assert jvp_np_from_gt != []

    assert np.allclose(flat_primals_np, flat_primals_np_from_gt)
    assert np.allclose(jvp_np, jvp_np_from_gt)


# write a test for vjp
@pytest.mark.parametrize(
    "x_", [[[[4.6, 2.1, 5], [2.8, 1.3, 6.2]], [[4.6, 2.1], [5, 2.8], [1.3, 6.2]]]]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("func_str", ["square", "cos"])
@pytest.mark.parametrize(
    "nested_structs", ["nested_input_nested_output", "nested_input_flat_output", "none"]
)
def test_vjp(x_, dtype, func_str, backend_fw, nested_structs):
    if backend_fw == "numpy":
        pytest.skip()

    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        base_func = ivy_backend.__dict__[func_str]
        if nested_structs == "none":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, ivy_backend, False, False
            )
            func = base_func
        elif nested_structs == "nested_input_nested_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, ivy_backend, True, True
            )
            func = base_func
        elif nested_structs == "nested_input_flat_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, ivy_backend, True, False
            )

            def func(x):
                return base_func(x["l"]["a"]) + base_func(x["l"]["b"])

        primals = (primals,)
        tangents = (tangents,)
        primals_out, vjp_fn = ivy_backend.vjp(func, *primals)
        vjp = vjp_fn(*tangents)
        flat_primals_np = helpers.flatten_and_to_np(ret=primals_out, backend=backend_fw)
        vjp_np = helpers.flatten_and_to_np(ret=vjp, backend=backend_fw)
        assert vjp_np != []

    with BackendHandler.update_backend("jax") as gt_backend:
        base_func = gt_backend.__dict__[func_str]
        if nested_structs == "none":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, gt_backend, False, False
            )
            func = base_func
        elif nested_structs == "nested_input_nested_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, gt_backend, True, True
            )
            func = base_func
        elif nested_structs == "nested_input_flat_output":
            primals, tangents = _get_primals_and_tangents(
                x_, dtype, gt_backend, True, False
            )

            def func(x):
                return base_func(x["l"]["a"]) + base_func(x["l"]["b"])

        primals = (primals,)
        tangents = (tangents,)
        primals_out_gt, vjp_fn = gt_backend.vjp(func, *primals)
        vjp = vjp_fn(*tangents)
        flat_primals_np_from_gt = helpers.flatten_and_to_np(
            ret=primals_out_gt, backend="jax"
        )
        vjp_np_from_gt = helpers.flatten_and_to_np(ret=vjp, backend="jax")
        assert vjp_np_from_gt != []

    assert np.allclose(flat_primals_np, flat_primals_np_from_gt)
    assert np.allclose(vjp_np, vjp_np_from_gt)
