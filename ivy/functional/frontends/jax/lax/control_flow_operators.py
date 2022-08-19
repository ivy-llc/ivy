# global
import ivy


def cond(pred, true_fun, false_fun, *operands, operand=None):
    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


def map(f, xs):
    return ivy.stack([f(x) for x in xs])
