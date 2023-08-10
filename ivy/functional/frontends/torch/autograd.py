# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import (
    handle_gradients,
)


def _add_grad(g_total, g):
    """Return g_total + g after checking None values."""
    if g is None:
        return g_total
    elif g_total is None:
        return g
    return g_total + g


def _create_grad_outputs(tensors, outputs=None):
    if tensors is None:
        ret = tuple()
        for out in outputs:
            ret += (torch_frontend.ones_like(out),)
        return ret
    elif isinstance(tensors, torch_frontend.Tensor):
        return (tensors,)
    return tuple(tensors)


def _grad_out_multiply(grad_out, jacobian_wrt_input):
    """
    return grad_out * jacobian_wrt_input after manipulating the shapes
    if grad_put is a 1-D tensor, this would be equivalent to:
    matmul(transpose(jacs), grad_out)
    """
    output_shape = grad_out.shape
    input_num_dims = len(jacobian_wrt_input.shape) - len(output_shape)
    expanded_grad_out = grad_out.view(output_shape + (1,) * input_num_dims)
    sum_dims = tuple(range(len(output_shape)))
    new_grad_out = (
        torch_frontend.sum(expanded_grad_out * jacobian_wrt_input, dim=sum_dims)
        if sum_dims
        else expanded_grad_out * jacobian_wrt_input
    )
    return new_grad_out


def _get_grad(output, input, grad_output):
    """Compute gradient of output w.r.t input."""

    # Case #1
    if output is input:
        return grad_output

    # Get inputs of the function that returned output
    func_inputs = output.func_inputs

    # Case #2
    # Reached end of graph. input & output are not connected
    if not func_inputs:
        return None

    # Case #3
    # Search for the input deeper in the graph
    grads = None

    # Jac function returns jacobians of all outputs of the function
    # We are only intersted in one of them
    if output.out_idx is None:
        jacs = handle_gradients(output.jac_fn)(func_inputs)
    else:
        jacs = ivy.index_nest(
            handle_gradients(output.jac_fn)(func_inputs), output.out_idx
        )

    all_indices = ivy.all_nested_indices(func_inputs)
    for idx in all_indices:
        func_input = ivy.index_nest(func_inputs, idx)
        jac_wrt_input = ivy.index_nest(jacs, idx)

        new_grad_out = _grad_out_multiply(grad_output, jac_wrt_input)
        grad = _get_grad(func_input, input, new_grad_out)
        grads = _add_grad(grads, grad)

    return grads


def _batched_get_grad(output, input, grad_output, batched):
    if batched:
        return torch_frontend.stack([_get_grad(output, input, g) for g in grad_output])
    return _get_grad(output, input, grad_output)


# @handle_gradients
def grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=None,
    create_graph=False,
    only_inputs=True,
    allow_unused=False,
    is_grads_batched=False,
):
    """Compute and return the sum of gradients of outputs with respect to each input."""
    inputs = (inputs,) if isinstance(inputs, torch_frontend.Tensor) else tuple(inputs)
    outputs = (
        (outputs,) if isinstance(outputs, torch_frontend.Tensor) else tuple(outputs)
    )
    grad_outputs = _create_grad_outputs(grad_outputs, outputs)

    ret = tuple()
    for input in inputs:
        if not input.requires_grad:
            raise RuntimeError("One of the input tensors does not require grad")

        grad_wrt_input = None
        for output, grad_output in zip(outputs, grad_outputs):
            if not output.requires_grad:
                raise RuntimeError("One of the output tensors does not require grad")

            g = _batched_get_grad(output, input, grad_output, is_grads_batched)
            grad_wrt_input = _add_grad(grad_wrt_input, g)

        if not allow_unused and grad_wrt_input is None:
            raise RuntimeError(
                "One of the differentiated Tensors appears to not have"
                " been used in the graph. Set allow_unused=True if this"
                " is the desired behavior."
            )
        ret += (grad_wrt_input,)
    return ret
