import torch


def is_native_sparse_array(x):
    return x.layout == torch.sparse_coo


def get_sparse_components(x):
    if x.layout == torch.sparse_coo:
        x = x.coalesce()
        return x.indices(), x.values(), x.size()
    raise Exception("not a Tensor with sparse COO layout")
