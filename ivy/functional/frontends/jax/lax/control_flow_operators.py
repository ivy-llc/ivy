"""Control flow operators for the implementation of Jax."""


def fori_loop(lower, upper, body_fun, init_val):
    """Implentation of for loop by upper and lower bound."""
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val
