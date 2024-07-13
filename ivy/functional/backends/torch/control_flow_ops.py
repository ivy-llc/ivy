def if_else(cond, body_fn, orelse_fn, vars):
    # back-compatibility
    if isinstance(cond, bool):
        v = cond

        def cond(*_):
            return v

    if callable(cond):
        cond = cond(**vars)
    else:
        cond = bool(cond)
    if cond:
        return body_fn(**vars)
    else:
        return orelse_fn(**vars)


def while_loop(test_fn, body_fn, vars):
    if isinstance(vars, dict):
        result = list(vars.values())
    else:
        result = list(vars)
    while test_fn(*result) is True:
        result = body_fn(*result)
        if not isinstance(result, tuple):
            result = (result,)
    return result
