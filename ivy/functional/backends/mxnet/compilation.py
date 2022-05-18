"""Collection of mxnet compilation functions."""

# global
import logging


# noinspection PyUnusedLocal
def compile(
    func, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None
):
    logging.warning(
        "MXnet does not support compiling arbitrary functions, consider writing a "
        "function using MXNet Symbolic backend instead for compiling.\n"
        "Now returning the unmodified function."
    )
    return func
