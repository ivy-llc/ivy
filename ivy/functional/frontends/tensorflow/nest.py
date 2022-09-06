# global
import ivy
import tensorflow as tf
import torch


def _is_composite_array(x):
    if isinstance(x, (tf.SparseTensor, tf.RaggedTensor)):
        return True
    if isinstance(x, torch.Tensor):
        if x.layout in [torch.sparse_coo, torch.sparse_csr]:
            return True
    return False


def _flatten_composite_array(x):
    if isinstance(x, tf.RaggedTensor):
        new_struc = [x.flat_values]
        for row_split in x.nested_row_splits:
            new_struc.append(row_split)
        return new_struc
    elif isinstance(x, tf.SparseTensor):
        return [x.indices, x.values, x.dense_shape]
    elif isinstance(x, torch.Tensor):
        if x.layout == torch.sparse_coo:
            x = x.coalesce()
            return [x.indices(), x.values(), ivy.native_array(x.size(), dtype="int64")]
        elif x.layout == torch.sparse_csr:
            return [
                x.crow_indices(),
                x.col_indices(),
                x.values(),
                ivy.native_array(x.size(), dtype="int64"),
            ]


def flatten(structure, expand_composites=False):
    if expand_composites and _is_composite_array(structure):
        return _flatten_composite_array(structure)
    elif isinstance(structure, (tuple, list)):
        new_struc = []
        for child in structure:
            new_struc += flatten(child)
        return new_struc
    elif isinstance(structure, dict):
        new_struc = []
        for key in sorted(structure):
            new_struc += flatten(structure[key])
        return new_struc
    return [structure]
