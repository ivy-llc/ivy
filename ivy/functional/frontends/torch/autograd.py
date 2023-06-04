# global
from typing import Union, Optional, Tuple, Sequence

# local
import ivy.functional.frontends.torch as torch_frontend

_TensorOrTensors = Union[torch_frontend.Tensor, Sequence[torch_frontend.Tensor]]


def _get_grad(output, input, grad_output):
    """Computes gradient of output w.r.t input."""
    # Get inputs of the function that returned output
    func_inputs = output.func_inputs

    # Reached end of graph. input & output are not connected
    if not func_inputs:
        return None

    grads = None
    for i, func_input in enumerate(func_inputs):
        if input is func_input:
            axis = list(range(len(output.shape)))
            grad = (output.grads[i] * grad_output).sum(dim=axis)
            if grads is not None:
                grads += (output.grads[i] * grad_output).sum(dim=axis)
            else:
                grads = (output.grads[i] * grad_output).sum(dim=axis)
            continue

        grad = _get_grad(func_input, input, grad_output)
        if grad is not None:
            axis = list(range(len(output.shape)))
            if grads is not None:
                grads += (output.grads[i].sum(dim=axis)) * grad
            else:
                grads = (output.grads[i].sum(dim=axis)) * grad

    return grads


def grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
    is_grads_batched: bool = False,
) -> Tuple[torch_frontend.Tensor, ...]:
    """Computes and returns the sum of gradients of outputs with respect to each
    input."""
    inputs = (inputs,) if isinstance(inputs, torch_frontend.Tensor) else inputs
    outputs = (outputs,) if isinstance(outputs, torch_frontend.Tensor) else outputs
    if grad_outputs is None:
        grad_outputs = [torch_frontend.ones_like(y) for y in outputs]
    else:
        grad_outputs = (
            (grad_outputs,)
            if isinstance(grad_outputs, torch_frontend.Tensor)
            else grad_outputs
        )

    ret = []
    for input in inputs:
        assert input.requires_grad, "One of the input tensors does not require grad"
        grad_wrt_input = None

        for output, grad_output in zip(outputs, grad_outputs):
            assert (
                output.requires_grad
            ), "One of the output tensors does not require grad"
            if is_grads_batched:
                g = torch_frontend.stack(
                    [_get_grad(output, input, g) for g in grad_output]
                )
            else:
                g = _get_grad(output, input, grad_output)

            if g is not None and grad_wrt_input is not None:
                grad_wrt_input += g
            elif grad_wrt_input is None:
                grad_wrt_input = g

        if not allow_unused:
            assert grad_wrt_input is not None, (
                "One of the differentiated Tensors appears                 to not have"
                " been used in the graph.                 Set allow_unused=True if this"
                " is the desired behavior."
            )
        ret += [grad_wrt_input]
    return tuple(ret)
