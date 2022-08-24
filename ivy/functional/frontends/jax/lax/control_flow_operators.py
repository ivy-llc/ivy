# global
import ivy


def cond(pred, true_fun, false_fun, *operands, operand=None, linear=None):
    if operand is not None:
        if operands:
            raise TypeError(
                "if `operand` is passed, positional `operands` should not be passed"
            )
        operands = (operand,)

    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def map(f, xs):
    return ivy.stack([f(x) for x in xs])


def switch(index, branches, *operands, operand=None):
    if operand is not None:
        if operands:
            raise TypeError(
                "if `operand` is passed, positional `operands` should not be passed"
            )
        operands = (operand,)

    index = max(index, 0)
    index = min(len(branches) - 1, index)
    return branches[index](*operands)
