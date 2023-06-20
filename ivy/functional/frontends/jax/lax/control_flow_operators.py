# global
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def associative_scan(fn, elems, reverse=False, axis=0):
    if not callable(fn):
        raise ivy.exceptions.IvyException("associative_scan: Argument fn should be callable.")

    if axis != 0:
        raise ivy.exceptions.IvyException("associative_scan: Axis other than 0 is not supported.")

    if reverse:
        elems = elems[::-1]

    acc = elems[0]
    results = [acc]
    for elem in elems[1:]:
        acc = fn(acc, elem)
        results.append(acc)

    if reverse:
        results = results[::-1]

    return ivy.stack(results, axis=axis)


@to_ivy_arrays_and_back
def map(f, xs):
    return ivy.stack([f(x) for x in xs])


@to_ivy_arrays_and_back
def switch(index, branches, *operands, operand=None):
    if operand is not None:
        if operands:
            raise ivy.utils.exceptions.IvyException(
                "if `operand` is passed, positional `operands` should not be passed"
            )
        operands = (operand,)

    index = max(index, 0)
    index = min(len(branches) - 1, index)
    return branches[index](*operands)


@to_ivy_arrays_and_back
def fori_loop(lower, upper, body_fun, init_val):
    if not (callable(body_fun)):
        raise ivy.exceptions.IvyException(
            "jax.lax.fori_loop: Argument body_fun should be callable."
        )
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


@to_ivy_arrays_and_back
def while_loop(cond_fun, body_fun, init_val):
    if not (callable(body_fun) and callable(cond_fun)):
        raise ivy.exceptions.IvyException(
            "jax.lax.while_loop: Arguments body_fun and cond_fun should be callable."
        )
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
