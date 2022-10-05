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
from math import pi, sin, sqrt


def _KBDW(window_length, beta, dtype=None):
    window_length = window_length // 2
    w = jnp.kaiser(window_length + 1, beta)
    sum_i_N = sum([w[i] for i in range(0, window_length + 1)])
    
    def sum_i_n(n):
        return sum([w[i] for i in range(0, n + 1)])
    dn_low = [sqrt(sum_i_n(i)/sum_i_N) for i in range(0, window_length)]
    
    def sum_2N_1_n(n):
        return sum([w[i] for i in range(0, 2 * window_length - n)])
    dn_mid = [sqrt(sum_2N_1_n(i)/sum_i_N) for i in range(window_length, 2*window_length)]
    
    return jnp.array(dn_low + dn_mid, dtype=dtype)

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
        "Jax does not support sparse array natively, None is returned for \
        indices, values and shape."
    )
    return None, None, None


def sinc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinc(x)


def vorbis_window(
    window_length: JaxArray,
    *,
    dtype:Optional[jnp.dtype] = jnp.float32,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.array([
        round(sin((pi/2)*(sin(pi*(i)/(window_length*2))**2)), 8)
        for i in range(1, window_length*2)[0::2]
    ], dtype=dtype)


def kaiser_bessel_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if periodic == True:
        return _KBDW(window_length + 1, beta, dtype)[:-1]
    else:
        return _KBDW(window_length, beta, dtype)