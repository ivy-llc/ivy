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
    # Sample a function
    fn = draw(st.sampled_from(["sin", "cos", "exp", "mean", "sum"]))

    # Generate function input
    dtypes, xs = draw(
        helpers.dtype_and_values(
            min_value=-20,
            max_value=20,
            max_num_dims=2,
            max_dim_size=2,
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

    return dtypes, xs, fn, is_grads_batched, g_output


@given(grad_args=_get_grad_args())
def test_torch_grad(grad_args):
    fw = ivy.current_backend_str()
    if fw == "paddle" or fw == "numpy":
        return
    if fw == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)

    dtypes, xs, fn, is_grads_batched, g_output = grad_args

    # Convert dtypes to native
    ivy_torch = ivy.with_backend("torch")
    dtypes_torch = [ivy_torch.as_native_dtype(t) for t in dtypes]

    # Main test
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

    # allow_unused=False
    with pytest.raises(Exception):
        torch_frontend.autograd.grad(y_ivy, x2_ivy)

    # Test with inputs with requires_grad = False
    with pytest.raises(Exception):
        x = torch_frontend.tensor(xs[0], dtype=dtypes[0])
        y = fn_ivy(x)
        torch_frontend.autograd.grad(y, x)

    # Test when input is used twice
    # test x = y
