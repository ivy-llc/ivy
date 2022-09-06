import torch


def is_native_sparse_array(x):
    # TODO: to add csr
    return x.layout == torch.sparse_coo


def init_data_sparse_array(indices, values, shape):
    return torch.sparse_coo_tensor(indices=indices.data, values=values.data, size=shape)


def init_native_components(x):
    if x.layout == torch.sparse_coo:
        x = x.coalesce()
        return x.indices(), x.values(), x.size()
    # TODO: to add csr
    raise Exception("not a Tensor with sparse COO layout")
