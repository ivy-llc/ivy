import tensorflow as tf


def if_else(cond, body_fn, orelse_fn, vars):
    # back-compatibility
    if isinstance(cond, bool):
        v = cond

        def cond(*_):
            return v

    cond = bool(cond(**vars))
    return tf.cond(cond, lambda: body_fn(**vars), lambda: orelse_fn(**vars))

    # use pythonic placeholder until the tracer supports callable arguments


def while_loop(test_fn, body_fn, vars):
    def body_fn_wrapper(*loop_vars):
        return body_fn(*loop_vars)

    def test_fn_wrapper(*loop_vars):
        return test_fn(*loop_vars)

    if not vars:
        vars = (0,)
    elif isinstance(vars, dict):
        vars = list(vars.values())
    return tf.while_loop(test_fn_wrapper, body_fn_wrapper, loop_vars=vars)


def for_loop(
    iterable,
    body_fn,
    vars,
):
    iterator = iterable.__iter__()

    vars_dict = _tuple_to_dict(vars)

    def test_fn(*args):
        nonlocal iterator, body_fn, vars_dict
        try:
            val = iterator.__next__()
        except StopIteration:
            return False

        vars_tuple = body_fn(val, _dict_to_tuple(vars_dict))

        for k in range(len(vars_tuple)):
            vars_dict[k] = vars_tuple[k]

        return True

    def empty_function(*args):
        return (0,)

    while_loop(test_fn, empty_function, ())

    return _dict_to_tuple(vars_dict)


def _tuple_to_dict(t):
    return {k: t[k] for k in range(len(t))}


def _dict_to_tuple(d):
    return tuple(d[k] for k in d)
