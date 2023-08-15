# global
from hypothesis import given
from hypothesis import strategies as st
import torch
import pytest

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import update_backend
import ivy.functional.frontends.torch as torch_frontend


@st.composite
def _get_grad_args(draw):
    # Case 1: functions with one input
    fn = draw(st.sampled_from(["sin", "cos", "square", "mean", "sum"]))
    dtypes, xs = draw(
        helpers.dtype_and_values(
            min_value=-20,
            max_value=20,
            max_num_dims=2,
            max_dim_size=2,
            available_dtypes=("float32", "float64"),
        )
    )

    # Case 2: functions with multi input
    fn2 = draw(st.sampled_from(["add", "matmul"]))
    dtypes2, xs2 = draw(
        helpers.dtype_and_values(
            min_value=-20,
            max_value=20,
            num_arrays=2,
            shape=(2, 2),
            shared_dtype=True,
            available_dtypes=("float32", "float64"),
        )
    )

    # Generate grad_outputs
    is_grads_batched = draw(st.booleans())
    ivy_fn = torch_frontend.__dict__[fn]
    out_shape = ivy_fn(xs[0]).shape

    if is_grads_batched:
        batch_size = draw(st.integers(min_value=1, max_value=2))
        g_out_shape = (batch_size,) + out_shape
    else:
        g_out_shape = out_shape

    _, g_output = draw(
        helpers.dtype_and_values(
            min_value=-20,
            max_value=20,
            dtype=dtypes,
            shape=g_out_shape,
        )
    )

    if len(out_shape) == 0 and not is_grads_batched:
        g_output = draw(st.sampled_from([g_output, None]))

    return dtypes, xs, fn, is_grads_batched, g_output, dtypes2, xs2, fn2


@given(grad_args=_get_grad_args())
def test_torch_grad(grad_args, backend_fw):
    if backend_fw == "numpy":
        pytest.skip()
    if backend_fw == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)

    dtypes, xs, fn, is_grads_batched, g_output, dtypes2, xs2, fn2 = grad_args
    ivy.set_backend(backend_fw)

    # Convert dtypes to native
    with update_backend("torch") as ivy_torch:
        dtypes_torch = [ivy_torch.as_native_dtype(t) for t in dtypes]
        dtypes_torch2 = [ivy_torch.as_native_dtype(t) for t in dtypes2]

    # Case 1: fn has single input and single output
    fn_ivy = torch_frontend.__dict__[fn]
    fn_torch = torch.__dict__[fn]

    x_ivy = torch_frontend.tensor(xs[0], requires_grad=True, dtype=dtypes[0])
    x_torch = torch.tensor(xs[0], requires_grad=True, dtype=dtypes_torch[0])

    y_ivy = fn_ivy(x_ivy)
    y_torch = fn_torch(x_torch)

    if g_output is not None:
        grad_outputs_ivy = torch_frontend.tensor(g_output[0], dtype=dtypes[0])
        grad_outputs_torch = torch.tensor(g_output[0], dtype=dtypes_torch[0])
    else:
        grad_outputs_ivy = None
        grad_outputs_torch = None

    grads_ivy = torch_frontend.autograd.grad(
        y_ivy,
        x_ivy,
        grad_outputs=grad_outputs_ivy,
        is_grads_batched=is_grads_batched,
    )

    grads_torch = torch.autograd.grad(
        y_torch,
        x_torch,
        grad_outputs=grad_outputs_torch,
        is_grads_batched=is_grads_batched,
    )

    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads_ivy[0], backend=backend_fw)

    with update_backend("torch") as ivy_torch:
        grads_flat_np_gt = helpers.flatten_and_to_np(
            backend="torch", ret=ivy_torch.to_ivy(grads_torch[0])
        )

    helpers.value_test(
        ret_np_flat=grads_flat_np,
        ret_np_from_gt_flat=grads_flat_np_gt,
        rtol=1e-4,
        atol=1e-4,
        backend=backend_fw,
    )

    # Multiple input
    fn_ivy2 = torch_frontend.__dict__[fn2]
    fn_torch2 = torch.__dict__[fn2]

    x_ivy1 = torch_frontend.tensor(xs2[0], requires_grad=True, dtype=dtypes2[0])
    x_ivy2 = torch_frontend.tensor(xs2[1], requires_grad=True, dtype=dtypes2[1])
    x_torch1 = torch.tensor(xs2[0], requires_grad=True, dtype=dtypes_torch2[0])
    x_torch2 = torch.tensor(xs2[1], requires_grad=True, dtype=dtypes_torch2[1])

    y_ivy2 = fn_ivy2(x_ivy1, x_ivy2)
    y_torch2 = fn_torch2(x_torch1, x_torch2)

    grad_outputs_ivy2 = torch_frontend.ones_like(y_ivy2)
    grad_outputs_torch2 = torch.ones_like(y_torch2)

    grads_ivy2 = torch_frontend.autograd.grad(
        y_ivy2,
        (x_ivy1, x_ivy2),
        grad_outputs=grad_outputs_ivy2,
    )

    grads_torch2 = torch.autograd.grad(
        y_torch2,
        (x_torch1, x_torch2),
        grad_outputs=grad_outputs_torch2,
    )

    with update_backend("torch") as ivy_torch:
        for g_ivy, g_torch in zip(grads_ivy2, grads_torch2):
            grads_flat_np = helpers.flatten_frontend_to_np(
                ret=g_ivy, backend=backend_fw
            )
            grads_flat_np_gt = helpers.flatten_and_to_np(
                backend="torch", ret=ivy_torch.to_ivy(g_torch)
            )

            helpers.value_test(
                ret_np_flat=grads_flat_np,
                ret_np_from_gt_flat=grads_flat_np_gt,
                rtol=1e-4,
                atol=1e-4,
                backend=backend_fw,
            )

    # Multiple outputs
    x_ivy = torch_frontend.tensor(xs2[0], requires_grad=True, dtype=dtypes2[0])
    x_torch = torch.tensor(xs2[0], requires_grad=True, dtype=dtypes_torch2[0])

    y_ivy = (x_ivy, torch_frontend.add(x_ivy, x_ivy))
    y_torch = (x_torch, torch.add(x_torch, x_torch))

    grad_outputs_ivy = (
        torch_frontend.ones_like(y_ivy[0]),
        torch_frontend.ones_like(y_ivy[1]),
    )
    grad_outputs_torch = (torch.ones_like(y_torch[0]), torch.ones_like(y_torch[1]))

    grads_ivy = torch_frontend.autograd.grad(
        y_ivy,
        x_ivy,
        grad_outputs=grad_outputs_ivy,
    )

    grads_torch = torch.autograd.grad(
        y_torch,
        x_torch,
        grad_outputs=grad_outputs_torch,
    )

    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads_ivy[0], backend=backend_fw)
    with update_backend("torch") as ivy_torch:
        grads_flat_np_gt = helpers.flatten_and_to_np(
            backend="torch", ret=ivy_torch.to_ivy(grads_torch[0])
        )

    helpers.value_test(
        ret_np_flat=grads_flat_np,
        ret_np_from_gt_flat=grads_flat_np_gt,
        rtol=1e-4,
        atol=1e-4,
        backend=backend_fw,
    )

    # Nested functions
    x_ivy1 = torch_frontend.tensor(xs2[0], requires_grad=True, dtype=dtypes2[0])
    x_ivy2 = torch_frontend.tensor(xs2[1], requires_grad=True, dtype=dtypes2[1])
    x_torch1 = torch.tensor(xs2[0], requires_grad=True, dtype=dtypes_torch2[0])
    x_torch2 = torch.tensor(xs2[1], requires_grad=True, dtype=dtypes_torch2[1])

    y_ivy2 = torch_frontend.mean(
        torch_frontend.matmul(
            torch_frontend.add(x_ivy1, x_ivy2), torch_frontend.sin(x_ivy2)
        )
    )
    y_torch2 = torch.mean(
        torch.matmul(
            torch.add(x_torch1, x_torch2),
            torch.sin(x_torch2),
        )
    )

    grads_ivy2 = torch_frontend.autograd.grad(y_ivy2, (x_ivy1, x_ivy2))
    grads_torch2 = torch.autograd.grad(y_torch2, (x_torch1, x_torch2))
    with update_backend("torch") as ivy_torch:
        for g_ivy, g_torch in zip(grads_ivy2, grads_torch2):
            grads_flat_np = helpers.flatten_frontend_to_np(
                ret=g_ivy, backend=backend_fw
            )
            grads_flat_np_gt = helpers.flatten_and_to_np(
                backend="torch", ret=ivy_torch.to_ivy(g_torch)
            )

            helpers.value_test(
                ret_np_flat=grads_flat_np,
                ret_np_from_gt_flat=grads_flat_np_gt,
                rtol=1e-4,
                atol=1e-4,
                backend=backend_fw,
            )

    # Test arg: allow_unused
    # Test inputs and outputs that are not
    # connected, allow_unused=True
    y_ivy = x_ivy
    x2_ivy = torch_frontend.tensor([1.0], requires_grad=True, dtype=dtypes[0])
    grads_ivy = torch_frontend.autograd.grad(y_ivy, x2_ivy, allow_unused=True)

    y_torch = x_torch
    x2_torch = torch.tensor([1.0], requires_grad=True, dtype=dtypes_torch[0])
    grads_torch = torch.autograd.grad(
        y_torch, x2_torch, grad_outputs=torch.ones_like(y_torch), allow_unused=True
    )

    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads_ivy[0], backend=backend_fw)
    with update_backend("torch") as ivy_torch:
        grads_flat_np_gt = helpers.flatten_and_to_np(
            backend="torch", ret=ivy_torch.to_ivy(grads_torch[0])
        )

    helpers.value_test(
        ret_np_flat=grads_flat_np,
        ret_np_from_gt_flat=grads_flat_np_gt,
        rtol=1e-4,
        atol=1e-4,
        backend=backend_fw,
    )

    # allow_unused=False
    with pytest.raises(Exception):
        torch_frontend.autograd.grad(y_ivy, x2_ivy)

    # Test inputs with requires_grad = False
    with pytest.raises(Exception):
        x = torch_frontend.tensor(xs[0], dtype=dtypes[0])
        y = fn_ivy(x)
        torch_frontend.autograd.grad(y, x)
