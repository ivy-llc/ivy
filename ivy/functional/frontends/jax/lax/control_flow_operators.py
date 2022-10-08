# global
import ivy
from ivy.functional.frontends.jax.func_wrapper import inputs_to_ivy_arrays


@inputs_to_ivy_arrays
def cond(pred, true_fun, false_fun, *operands, operand=None, linear=None):
    if operand is not None:
        if operands:
            raise ivy.exceptions.IvyException(
                "if `operand` is passed, positional `operands` should not be passed"
            )
        operands = (operand,)

    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


@inputs_to_ivy_arrays
def map(f, xs):
    return ivy.stack([f(x) for x in xs])


@inputs_to_ivy_arrays
def switch(index, branches, *operands, operand=None):
    if operand is not None:
        if operands:
            raise ivy.exceptions.IvyException(
                "if `operand` is passed, positional `operands` should not be passed"
            )
        operands = (operand,)

    index = max(index, 0)
    index = min(len(branches) - 1, index)
    return branches[index](*operands)
