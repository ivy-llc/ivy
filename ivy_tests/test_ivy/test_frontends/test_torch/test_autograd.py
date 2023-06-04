# global
from hypothesis import given
from hypothesis import strategies as st
import torch
import pytest

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.frontends.torch as torch_frontend


@st.composite
def _get_grad_args(draw):
    dtypes, xs = draw(
        helpers.dtype_and_values(
            min_value=-20,
            max_value=20,
            num_arrays=3,
            shared_dtype=True,
            available_dtypes=("float32", "float64"),
        )
    )
    shape = xs[0].shape
    batch_size = draw(st.integers(min_value=1, max_value=3))
    g_out_shape = tuple([batch_size] + list(shape))
    g_dtypes, g_output = draw(
        helpers.dtype_and_values(
            min_value=-20,
            max_value=20,
            dtype=dtypes,
            shape=g_out_shape,
            available_dtypes=("float32", "float64"),
        )
    )
    return dtypes, xs, g_dtypes, g_output


@given(xs_grad_outputs=_get_grad_args())
def test_torch_grad(xs_grad_outputs):
    fw = ivy.current_backend_str()
    if fw == "paddle" or fw == "numpy":
        return
    if fw == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)

    dtypes, xs, g_dtypes, batched_grad_outputs = xs_grad_outputs
    ivy_torch = ivy.with_backend("torch")
    dtypes_torch = [ivy_torch.as_native_dtype(t) for t in dtypes]
    g_dtypes_torch = [ivy_torch.as_native_dtype(t) for t in g_dtypes]
    # TODO: when y=x

    # Test single in single out
    fn_ivy = lambda x: torch_frontend.exp(torch_frontend.cos(x))
    x_ivy = torch_frontend.tensor(xs[0], requires_grad=True, dtype=dtypes[0])
    y_ivy = fn_ivy(x_ivy)
    grads_ivy = torch_frontend.autograd.grad(y_ivy, x_ivy, retain_graph=True)

    fn_torch = lambda x: torch.exp(torch.cos(x))
    x_torch = torch.tensor(xs[0], requires_grad=True, dtype=dtypes_torch[0])
    y_torch = fn_torch(x_torch)
    grads_torch = torch.autograd.grad(
        y_torch, x_torch, grad_outputs=torch.ones_like(y_torch), retain_graph=True
    )

    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads_ivy[0])
    ivy.set_backend("torch")
    grads_flat_np_gt = helpers.flatten_and_to_np(ret=ivy.to_ivy(grads_torch[0]))
    ivy.previous_backend()

    helpers.value_test(
        ret_np_flat=grads_flat_np,
        ret_np_from_gt_flat=grads_flat_np_gt,
        rtol=1e-4,
        atol=1e-4,
    )

    # Test many inputs & outputs
    x2_ivy = torch_frontend.tensor(xs[1], requires_grad=True, dtype=dtypes[0])
    y2_ivy = fn_ivy(x2_ivy)
    grads_ivy = torch_frontend.autograd.grad([y_ivy, y2_ivy], [x_ivy, x2_ivy])

    x2_torch = torch.tensor(xs[1], requires_grad=True, dtype=dtypes_torch[0])
    y2_torch = fn_torch(x2_torch)
    grads_torch = torch.autograd.grad(
        [y_torch, y2_torch],
        [x_torch, x2_torch],
        grad_outputs=[torch.ones_like(y2_torch)] * 2,
    )

    for g1, g2 in zip(grads_ivy, grads_torch):
        grads_flat_np = helpers.flatten_frontend_to_np(ret=g1)
        ivy.set_backend("torch")
        grads_flat_np_gt = helpers.flatten_and_to_np(ret=ivy.to_ivy(g2))
        ivy.previous_backend()
        helpers.value_test(
            ret_np_flat=grads_flat_np,
            ret_np_from_gt_flat=grads_flat_np_gt,
            rtol=1e-4,
            atol=1e-4,
        )

    # Test inputs and outputs that are not connected
    y_ivy = x_ivy
    grads_ivy = torch_frontend.autograd.grad(y_ivy, x2_ivy, allow_unused=True)
    y_torch = x_torch
    grads_torch = torch.autograd.grad(
        y_torch, x2_torch, grad_outputs=torch.ones_like(y_torch), allow_unused=True
    )

    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads_ivy[0])
    ivy.set_backend("torch")
    grads_flat_np_gt = helpers.flatten_and_to_np(ret=ivy.to_ivy(grads_torch[0]))
    ivy.previous_backend()
    helpers.value_test(
        ret_np_flat=grads_flat_np,
        ret_np_from_gt_flat=grads_flat_np_gt,
        rtol=1e-4,
        atol=1e-4,
    )

    # Test arg: allow_unused
    y_ivy = x_ivy
    with pytest.raises(Exception):
        torch_frontend.autograd.grad(y_ivy, x2_ivy)

    # Test with inputs with requires_grad = False
    with pytest.raises(Exception):
        x = torch_frontend.tensor(xs[0], dtype=dtypes[0])
        y = fn_ivy(x)
        torch_frontend.autograd.grad(y, x)

    # Test arg: grad_outputs
    x_ivy = torch_frontend.tensor(xs[0], requires_grad=True, dtype=dtypes[0])
    y_ivy = fn_ivy(x_ivy)
    grad_outputs_ivy = torch_frontend.tensor(xs[2])
    grads_ivy = torch_frontend.autograd.grad(
        y_ivy,
        x_ivy,
        grad_outputs=grad_outputs_ivy,
    )

    x_torch = torch.tensor(xs[0], requires_grad=True, dtype=dtypes_torch[0])
    y_torch = fn_torch(x_torch)
    grad_outputs_torch = torch.tensor(xs[2])
    grads_torch = torch.autograd.grad(
        y_torch,
        x_torch,
        grad_outputs=grad_outputs_torch,
    )

    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads_ivy[0])
    ivy.set_backend("torch")
    grads_flat_np_gt = helpers.flatten_and_to_np(ret=ivy.to_ivy(grads_torch[0]))
    ivy.previous_backend()
    helpers.value_test(
        ret_np_flat=grads_flat_np,
        ret_np_from_gt_flat=grads_flat_np_gt,
        rtol=1e-4,
        atol=1e-4,
    )

    # Test arg: is_grads_batched
    x_ivy = torch_frontend.tensor(xs[0], requires_grad=True, dtype=dtypes[0])
    y_ivy = fn_ivy(x_ivy)
    grad_outputs_ivy = torch_frontend.tensor(batched_grad_outputs[0], dtype=g_dtypes[0])
    grads_ivy = torch_frontend.autograd.grad(
        y_ivy,
        x_ivy,
        grad_outputs=grad_outputs_ivy,
        is_grads_batched=True,
    )

    x_torch = torch.tensor(xs[0], requires_grad=True, dtype=dtypes_torch[0])
    y_torch = fn_torch(x_torch)
    grad_outputs_torch = torch.tensor(batched_grad_outputs[0], dtype=g_dtypes_torch[0])
    grads_torch = torch.autograd.grad(
        y_torch,
        x_torch,
        grad_outputs=grad_outputs_torch,
        is_grads_batched=True,
    )

    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads_ivy[0])
    ivy.set_backend("torch")
    grads_flat_np_gt = helpers.flatten_and_to_np(ret=ivy.to_ivy(grads_torch[0]))
    ivy.previous_backend()
    helpers.value_test(
        ret_np_flat=grads_flat_np,
        ret_np_from_gt_flat=grads_flat_np_gt,
        rtol=1e-4,
        atol=1e-4,
    )
    # Test when input is used twice
    # TODO: Wait for ivy.jac
