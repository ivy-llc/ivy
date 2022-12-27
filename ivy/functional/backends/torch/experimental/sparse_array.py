import ivy
from ivy.functional.experimental.sparse_array import (
    _verify_coo_components,
    _verify_csr_components,
    _verify_csc_components,
    _is_data_not_indices_values_and_shape,
    _is_coo,
    _is_csr,
)
import torch


def is_native_sparse_array(x):
    return x.layout in [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc]


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
        return torch.sparse_coo_tensor(
            indices=coo_indices, values=values, size=dense_shape
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
        return torch.sparse_csr_tensor(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            size=dense_shape,
        )
    else:
        _verify_csc_components(
            ccol_indices=csc_ccol_indices,
            row_indices=csc_row_indices,
            values=values,
            dense_shape=dense_shape,
        )
        return torch.sparse_csc_tensor(
            ccol_indices=csc_ccol_indices,
            row_indices=csc_row_indices,
            values=values,
            size=dense_shape,
        )


def native_sparse_array_to_indices_values_and_shape(x):
    if x.layout == torch.sparse_coo:
        x = x.coalesce()
        return {"indices": x.indices()}, x.values(), x.size()
    elif x.layout == torch.sparse_csr:
        return (
            {"crow_indices": x.crow_indices(), "col_indices": x.col_indices()},
            x.values(),
            x.size(),
        )
    elif x.layout == torch.sparse_csc:
        return (
            {"ccol_indices": x.ccol_indices(), "row_indices": x.row_indices()},
            x.values(),
            x.size(),
        )
    raise ivy.exceptions.IvyException("not a sparse COO/CSR/CSC Tensor")
