from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    handle_array_like_without_promotion,
    to_native_arrays_and_back,
    to_ivy_arrays_and_back,
)


def IfElse(cond, body_fn, orelse_fn, vars):

    body_fn = to_ivy_arrays_and_back(body_fn)
    orelse_fn = to_ivy_arrays_and_back(orelse_fn)

    return if_else(cond, body_fn, orelse_fn, vars)


@to_native_arrays_and_back
@handle_array_like_without_promotion
def if_else(cond, body_fn, orelse_fn, vars):
    return current_backend().if_else(cond, body_fn, orelse_fn, vars)


def WhileLoop(test_fn, body_fn, vars):

    test_fn = to_ivy_arrays_and_back(test_fn)
    body_fn = to_ivy_arrays_and_back(body_fn)

    return while_loop(test_fn, body_fn, vars)


@to_native_arrays_and_back
@handle_array_like_without_promotion
def while_loop(test_fn, body_fn, vars):
    return current_backend().while_loop(test_fn, body_fn, vars)
