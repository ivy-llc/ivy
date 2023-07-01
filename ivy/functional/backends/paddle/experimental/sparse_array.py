import ivy
from ivy.functional.ivy.experimental.sparse_array import (
    _verify_coo_components,
    _verify_csr_components,
    _is_data_not_indices_values_and_shape,
)
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
)
from ivy.utils.exceptions import IvyNotImplementedException
import paddle

# local
from .. import backend_version


def is_native_sparse_array(x: paddle.Tensor) -> bool:
    return x.is_sparse_coo() or x.is_sparse_csr()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("int8",)}}, backend_version
)
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
) -> paddle.Tensor:
    format = format.lower()

    if format not in ["coo", "csr"]:
        raise IvyNotImplementedException(
            "paddle only supports 'coo' and 'csr' sparse formats."
        )

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

    if format == "coo":
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
        return paddle.sparse.sparse_coo_tensor(
            indices=coo_indices,
            values=values,
            shape=dense_shape,
            dtype=dtype,
            place=device,
            stop_gradient=not requires_grad,
        )
    else:
        _verify_csr_components(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            dense_shape=dense_shape,
        )
        return paddle.sparse.sparse_csr_tensor(
            crows=crow_indices,
            cols=col_indices,
            values=values,
            shape=dense_shape,
            dtype=dtype,
            place=device,
            stop_gradient=not requires_grad,
        )


def native_sparse_array_to_indices_values_and_shape(x):
    if not is_native_sparse_array(x):
        raise ivy.utils.exceptions.IvyException("not a Paddle Sparse Array")
    if x.is_sparse_coo():
        return {"coo_indices": x.indices()}, x.values(), x.shape
    else:
        return (
            {"crow_indices": x.crows(), "col_indices": x.cols()},
            x.values(),
            x.shape,
        )
