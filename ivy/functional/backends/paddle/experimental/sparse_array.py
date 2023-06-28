from ivy.utils.exceptions import IvyNotImplementedException
import paddle


def is_native_sparse_array(x: paddle.Tensor) -> bool:
    return x.is_sparse_coo() or x.is_sparse_csr()


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
    raise IvyNotImplementedException()


def native_sparse_array_to_indices_values_and_shape(x):
    raise IvyNotImplementedException()
