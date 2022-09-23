import logging
import ivy
import jax
from ivy.functional.backends.jax import JaxArray
from typing import Optional
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_coo_not_csr,
)


def is_native_sparse_array(x):
    """Jax does not support sparse arrays natively."""
    return False


def native_sparse_array(
        data=None,
        *,
        coo_indices=None,
        csr_crow_indices=None,
        csr_col_indices=None,
        values=None,
        dense_shape=None
):
    ivy.assertions.check_exists(
        data,
        inverse=True,
        message="data cannot be specified, Jax does not support sparse array natively",
    )
    if _is_coo_not_csr(
            coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
    else:
        _verify_csr_components(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            dense_shape=dense_shape,
        )
    logging.warning("Jax does not support sparse array natively, None is returned.")
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    logging.warning(
        "Jax does not support sparse array natively, None is returned for \
        indices, values and shape."
    )
    return None, None, None


def ifft(input: JaxArray, n: Optional[int] = None,
         dim: Optional[int] = None, axis: Optional[int] = None,
         norm: Optional[str] = None, name: Optional[str] = None):
    if axis is None:
        axis = -1
    return jax.numpy.ifft(a=input, n=n, axis=axis, norm=norm)
