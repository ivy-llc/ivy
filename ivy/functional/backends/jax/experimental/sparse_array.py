import logging
import ivy
from ivy.functional.experimental.sparse_array import (
    _verify_coo_components,
    _verify_csr_components,
    _verify_csc_components,
    _is_coo,
    _is_csr,
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
    csc_ccol_indices=None,
    csc_row_indices=None,
    values=None,
    dense_shape=None,
):
    ivy.assertions.check_exists(
        data,
        inverse=True,
        message="data cannot be specified, Jax does not support sparse array natively",
    )
    if _is_coo(
        coo_indices,
        csr_crow_indices,
        csr_col_indices,
        csc_ccol_indices,
        csc_row_indices,
        values,
        dense_shape,
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
    elif _is_csr(
        coo_indices,
        csr_crow_indices,
        csr_col_indices,
        csc_ccol_indices,
        csc_row_indices,
        values,
        dense_shape,
    ):
        _verify_csr_components(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            dense_shape=dense_shape,
        )
    else:
        _verify_csc_components(
            ccol_indices=csc_ccol_indices,
            row_indices=csc_row_indices,
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
