# global
import tensorflow as tf


def flatten(structure, expand_composites=False):
    if expand_composites and isinstance(structure, tf.RaggedTensor):
        new_struc = [structure.flat_values]
        for row_split in structure.nested_row_splits:
            new_struc.append(row_split)
        return new_struc
    if expand_composites and isinstance(structure, tf.SparseTensor):
        return [structure.indices, structure.values, structure.dense_shape]
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
