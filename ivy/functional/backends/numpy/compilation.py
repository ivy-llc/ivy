"""Collection of Numpy compilation functions."""

# global
import logging


# noinspection PyUnusedLocal
def compile(
    func, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None
):
    logging.warning(
        "Numpy does not support compiling functions.\n"
        "Now returning the unmodified function."
    )
    return func
