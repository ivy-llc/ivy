import ivy
from ivy.functional.ivy.experimental.sparse_array import (
    _verify_bsr_components,
    _verify_bsc_components,
    _verify_coo_components,
    _verify_csr_components,
    _verify_csc_components,
    _is_data_not_indices_values_and_shape,
)
import torch


def is_native_sparse_array(x):
    return x.layout in [
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsc,
        torch.sparse_csc,
    ]


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
    ):
        ivy.utils.assertions.check_true(
            ivy.is_native_sparse_array(data), message="not a sparse array"
        )
        return data

    format = format.lower()

    if format == "coo":
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
        return torch.sparse_coo_tensor(
            indices=coo_indices, values=values, size=dense_shape
        )
    elif format == "csr":
        _verify_csr_components(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            dense_shape=dense_shape,
        )
        return torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=dense_shape,
        )
    elif format == "csc":
        _verify_csc_components(
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=values,
            dense_shape=dense_shape,
        )
        return torch.sparse_csc_tensor(
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=values,
            size=dense_shape,
        )
    elif format == "bsc":
        _verify_bsc_components(
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=values,
            dense_shape=dense_shape,
        )
    else:
        _verify_bsr_components(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            dense_shape=dense_shape,
        )


def native_sparse_array_to_indices_values_and_shape(x):
    if x.layout == torch.sparse_coo:
        x = x.coalesce()
        return {"coo_indices": x.indices()}, x.values(), x.size()
    elif x.layout == torch.sparse_csr or x.layout == torch.sparse_bsr:
        return (
            {"crow_indices": x.crow_indices(), "col_indices": x.col_indices()},
            x.values(),
            x.size(),
        )
    elif x.layout == torch.sparse_bsc or x.layout == torch.sparse_csc:
        return (
            {"ccol_indices": x.crow_indices(), "row_indices": x.col_indices()},
            x.values(),
            x.size(),
        )
    raise ivy.utils.exceptions.IvyException("not a sparse COO/CSR/CSC/BSC/BSR Tensor")
