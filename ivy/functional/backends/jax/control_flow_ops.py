import jax


def if_else(cond, body_fn, orelse_fn, vars):
    # back-compatibility
    if isinstance(cond, bool):
        v = cond
        cond = lambda *_: v

    cond = bool(cond(*vars))
    with jax.disable_jit():
        final_vars = jax.lax.cond(
            cond, lambda: body_fn(*vars), lambda: orelse_fn(*vars)
        )
    return final_vars


def while_loop(test_fn, body_fn, vars):
    def body_fn_wrapper(loop_vars):
        return body_fn(*loop_vars)

    def test_fn_wrapper(loop_vars):
        return test_fn(*loop_vars)

    with jax.disable_jit():
        final_loop_vars = jax.lax.while_loop(test_fn_wrapper, body_fn_wrapper, vars)
    return final_loop_vars
