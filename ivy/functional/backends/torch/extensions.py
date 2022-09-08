import ivy
from ivy.functional.ivy.extensions import _verify_coo_components
import torch


def is_native_sparse_array(x):
    # TODO: to add csr
    return x.layout == torch.sparse_coo


def native_sparse_array(data=None, *, indices=None, values=None, dense_shape=None):
    if data is not None:
        if ivy.exists(indices) or ivy.exists(values) or ivy.exists(dense_shape):
            raise Exception(
                "only specify either data or or all three components: \
                indices, values and dense_shape"
            )
        assert ivy.is_native_sparse_array(data), "not a sparse array"
        return data
    _verify_coo_components(indices=indices, values=values, dense_shape=dense_shape)
    return torch.sparse_coo_tensor(indices=indices, values=values, size=dense_shape)


def native_sparse_array_to_indices_values_and_shape(x):
    if x.layout == torch.sparse_coo:
        x = x.coalesce()
        return x.indices(), x.values(), x.size()
    # TODO: to add csr
    raise Exception("not a Tensor with sparse COO layout")
