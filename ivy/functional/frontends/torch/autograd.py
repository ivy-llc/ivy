# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import (
    handle_gradients,
    to_ivy_arrays_and_back,
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


def _get_output(outputs, out_idx):
    # Function might return multiple outputs
    # We are only intersted in one of them
    if out_idx is None:
        return outputs
    return ivy.index_nest(outputs, out_idx)


def _get_wrapped_fn(output, input):
    """Compute gradient of output w.r.t input."""

    # Case #1
    if not isinstance(output, torch_frontend.Tensor):
        return None
    # Case #2
    if output is input:
        return lambda x: x

    # Get inputs of the function that returned output
    func_inputs = output.func_inputs
    out_func = output._func

    # Case #3
    # Reached end of graph. input & output are not connected
    if not func_inputs:
        return None

    # Case #4
    # Search for the input deeper in the graph
    in_funcs = []
    in_idxs = []
    all_indices = ivy.all_nested_indices(func_inputs)
    for idx in all_indices:
        func_input = ivy.index_nest(func_inputs, idx)
        f = _get_wrapped_fn(func_input, input)
        if f is not None:
            in_funcs += [f]
            in_idxs += [idx]

    if len(in_funcs) == 0:
        return None

    def wrapped_fn(x):
        func_inputs_mutable = ivy.copy_nest(func_inputs, to_mutable=True)

        for idx, in_func in zip(in_idxs, in_funcs):
            y = in_func(x)
            ivy.set_nest_at_index(func_inputs_mutable, idx, y)
        return _get_output(
            out_func(*func_inputs_mutable[0], **func_inputs_mutable[1]), output.out_idx
        )

    return wrapped_fn


def to_frontend_array_and_back(fn):
    def _to_frontend_array_and_back(x):
        x_torch = torch_frontend.Tensor(x, _init_overload=True, requires_grad=True)
        ret = fn(x_torch)
        return ret.ivy_array

    return _to_frontend_array_and_back


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


def _get_grad(wrapped_fn, input, grad_output):
    jac_fn = to_ivy_arrays_and_back(ivy.jac(to_frontend_array_and_back(wrapped_fn)))
    jacs = jac_fn(input)
    return _grad_out_multiply(grad_output, jacs)


def _batched_get_grad(wrapped_fn, input, grad_output, batched):
    if batched:
        return torch_frontend.stack(
            [_get_grad(wrapped_fn, input, g) for g in grad_output]
        )
    return _get_grad(wrapped_fn, input, grad_output)


@handle_gradients
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

            wrapped_fn = _get_wrapped_fn(output, input)
            g = None
            if wrapped_fn is not None:
                g = _batched_get_grad(wrapped_fn, input, grad_output, is_grads_batched)
            grad_wrt_input = _add_grad(grad_wrt_input, g)

        if not allow_unused and grad_wrt_input is None:
            raise RuntimeError(
                "One of the differentiated Tensors appears to not have"
                " been used in the graph. Set allow_unused=True if this"
                " is the desired behavior."
            )
        ret += (grad_wrt_input,)
    return ret
