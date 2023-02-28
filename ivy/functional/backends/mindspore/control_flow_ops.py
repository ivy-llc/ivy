def if_else(cond, body_fn, orelse_fn, vars):
    if cond:
        return body_fn(*vars)
    else:
        return orelse_fn(*vars)


def while_loop(test_fn, body_fn, vars):
    result = vars
    while test_fn(*result):
        result = body_fn(*result)
    return result
