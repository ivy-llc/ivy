import logging
from typing import Optional
import ivy
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_coo_not_csr,
)
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp
import math


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
    dense_shape=None,
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
        "Jax does not support sparse array natively, None is returned for        "
        " indices, values and shape."
    )
    return None, None, None


def sinc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinc(x)


def vorbis_window(
    window_length: JaxArray,
    *,
    dtype: Optional[jnp.dtype] = jnp.float32,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.array(
        [
            round(
                math.sin(
                    (ivy.pi / 2) * (math.sin(ivy.pi * (i) / (window_length * 2)) ** 2)
                ),
                8,
            )
            for i in range(1, window_length * 2)[0::2]
        ],
        dtype=dtype,
    )
