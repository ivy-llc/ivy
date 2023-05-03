# global
import logging

# local
import ivy
from ivy.functional.ivy.experimental.sparse_array import (
    _is_valid_format,
    _verify_bsc_components,
    _verify_bsr_components,
    _verify_coo_components,
    _verify_csc_components,
    _verify_csr_components,
)


def is_native_sparse_array(x):
    """Numpy does not support sparse arrays natively."""
    return False


def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    crow_indices=None,
    col_indices=None,
    ccol_indices=None,
    row_indices=None,
    values=None,
    dense_shape=None,
    format="coo",
):
    ivy.utils.assertions.check_exists(
        data,
        inverse=True,
        message=(
            "data cannot be specified, Numpy does not support sparse array natively"
        ),
    )

    if not _is_valid_format(
        coo_indices,
        crow_indices,
        col_indices,
        ccol_indices,
        row_indices,
        values,
        dense_shape,
        format,
    ):
        raise ivy.utils.exceptions.IvyException(
            "format should be one of the strings coo, csr, csc, bsr, and bsc."
        )

    format = format.lower()

    if format == "coo":
        _verify_coo_components(
            indices=coo_indices,
            values=values,
            dense_shape=dense_shape,
        )
    elif format == "csr":
        _verify_csr_components(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            dense_shape=dense_shape,
        )
    elif format == "bsr":
        _verify_bsr_components(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            dense_shape=dense_shape,
        )
    elif format == "csc":
        _verify_csc_components(
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=values,
            dense_shape=dense_shape,
        )
    else:
        _verify_bsc_components(
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=values,
            dense_shape=dense_shape,
        )
    logging.warning("Numpy does not support sparse array natively, None is returned.")
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    logging.warning(
        "Numpy does not support sparse array natively, None is returned for        "
        " indices, values and shape."
    )
    return None, None, None
