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


def while_loop(cond_fun, body_fun, init_val):
    # compile
    cond = ivy.compile(cond_fun, example_inputs=[init_val])
    body = ivy.compile(body_fun, example_inputs=[init_val])
    # function body
    val = init_val
    while ivy.all(cond(val)):
        val = body(val)
    return val
