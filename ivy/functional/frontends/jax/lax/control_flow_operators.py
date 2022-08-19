# global
import ivy


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
