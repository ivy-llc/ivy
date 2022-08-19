# global
import ivy
import tensorflow as tf
import torch


def _is_sparse_array(x):
    if isinstance(x, tf.SparseTensor):
        return True
    if isinstance(x, torch.Tensor):
        if x.layout in [torch.sparse_coo, torch.sparse_csr]:
            return True
    return False


def _flatten_sparse_array(x):
    if isinstance(x, tf.SparseTensor):
        return [x.indices, x.values, x.dense_shape]
    if isinstance(x, torch.Tensor):
        if x.layout == torch.sparse_coo:
            x = x.coalesce()
            return [x.indices(), x.values(), ivy.native_array(x.size(), dtype="int64")]
        if x.layout == torch.sparse_csr:
            return [
                x.crow_indices(),
                x.col_indices(),
                x.values(),
                ivy.native_array(x.size(), dtype="int64"),
            ]


def flatten(structure, expand_composites=False):
    if expand_composites:
        if isinstance(structure, tf.RaggedTensor):
            new_struc = [structure.flat_values]
            for row_split in structure.nested_row_splits:
                new_struc.append(row_split)
            return new_struc
        if _is_sparse_array(structure):
            return _flatten_sparse_array(structure)
    if isinstance(structure, (tuple, list)):
        new_struc = []
        for child in structure:
            new_struc += flatten(child)
        return new_struc
    if isinstance(structure, dict):
        new_struc = []
        for key in sorted(structure):
            new_struc += flatten(structure[key])
        return new_struc
    return [structure]
