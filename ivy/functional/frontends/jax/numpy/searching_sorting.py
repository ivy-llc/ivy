# global
import logging

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def argmax(a, axis=None, out=None, keepdims=False):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def argwhere(a, /, *, size=None, fill_value=None):
    if size is None and fill_value is None:
        return ivy.argwhere(a)

    result = ivy.matrix_transpose(
        ivy.vstack(ivy.nonzero(a, size=size, fill_value=fill_value))
    )
    num_of_dimensions = a.ndim

    if num_of_dimensions == 0:
        return result[:0].reshape(result.shape[0], 0)

    return result.reshape(result.shape[0], num_of_dimensions)


@to_ivy_arrays_and_back
def argsort(a, axis=-1, kind="stable", order=None):
    if kind != "stable":
        logging.warning(
            "'kind' argument to argsort is ignored; only 'stable' sorts "
            "are supported."
        )
    if order is not None:
        raise ivy.exceptions.IvyError("'order' argument to argsort is not supported.")

    return ivy.argsort(a, axis=axis)


@to_ivy_arrays_and_back
def msort(a):
    return ivy.msort(a)
