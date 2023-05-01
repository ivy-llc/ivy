# global
import tensorflow as tf
import logging

# local
import ivy
from ivy.functional.ivy.experimental.sparse_array import (
    _is_data_not_indices_values_and_shape,
    _verify_bsc_components,
    _verify_bsr_components,
    _verify_coo_components,
    _verify_csc_components,
    _verify_csr_components,
)


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)


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
    if _is_data_not_indices_values_and_shape(
        data,
        coo_indices,
        crow_indices,
        col_indices,
        ccol_indices,
        row_indices,
        values,
        dense_shape,
        format,
    ):
        ivy.utils.assertions.check_true(
            ivy.is_native_sparse_array(data), message="not a sparse array"
        )
        return data

    format = format.lower()

    if format == "coo":
        _verify_coo_components(
            coo_indices,
            values,
            dense_shape,
        )
        all_coordinates = []
        for i in range(values.shape[0]):
            coordinate = ivy.gather(coo_indices, ivy.array([[i]]))
            coordinate = ivy.reshape(coordinate, (coo_indices.shape[0],))
            all_coordinates.append(coordinate.to_list())
        return tf.SparseTensor(
            indices=all_coordinates, values=values, dense_shape=dense_shape
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

    logging.warning(
        f"Tensorflow does not support {format.upper()} \
sparse array natively. None is returned."
    )
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    if isinstance(x, tf.SparseTensor):
        return {"coo_indices": x.indices}, x.values, x.dense_shape
    raise ivy.utils.exceptions.IvyException("not a SparseTensor")
