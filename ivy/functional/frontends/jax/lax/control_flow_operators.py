# global
import ivy


def cond(pred, true_fun, false_fun, *operands, operand=None, linear=None):
    if operand is not None and operands:
        raise TypeError(
            "if `operand` is passed, positional `operands` should not be passed"
        )
    operands = (operand,)

    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


def map(f, xs):
    return ivy.stack([f(x) for x in xs])
