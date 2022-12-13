# global
import tensorflow as tf
import logging

# local
import ivy
from ivy.functional.experimental.sparse_array import (
    _verify_coo_components,
    _verify_csr_components,
    _verify_csc_components,
    _is_data_not_indices_values_and_shape,
    _is_coo,
    _is_csr,
)


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)


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
    if _is_data_not_indices_values_and_shape(
        data,
        coo_indices,
        csr_crow_indices,
        csr_col_indices,
        csc_ccol_indices,
        csc_row_indices,
        values,
        dense_shape,
    ):
        ivy.assertions.check_true(
            ivy.is_native_sparse_array(data), message="not a sparse array"
        )
        return data
    elif _is_coo(
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
        all_coordinates = []
        for i in range(values.shape[0]):
            coordinate = ivy.gather(coo_indices, ivy.array([[i]]))
            coordinate = ivy.reshape(coordinate, (coo_indices.shape[0],))
            all_coordinates.append(coordinate.to_list())
        return tf.SparseTensor(
            indices=all_coordinates, values=values, dense_shape=dense_shape
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
        logging.warning(
            "Tensorflow does not support CSR sparse array natively. None is returned."
        )
    else:
        _verify_csc_components(
            ccol_indices=csc_ccol_indices,
            row_indices=csc_row_indices,
            values=values,
            dense_shape=dense_shape,
        )
        logging.warning(
            "Tensorflow does not support CSC sparse array natively. None is returned."
        )
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    if isinstance(x, tf.SparseTensor):
        return {"indices": x.indices}, x.values, x.dense_shape
    raise ivy.exceptions.IvyException("not a SparseTensor")
