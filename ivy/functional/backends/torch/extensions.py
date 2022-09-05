import torch


def is_native_sparse_array(x):
    return x.layout in [torch.sparse_coo, torch.sparse_csr]
