# global
import ivy

try:
    import tensorflow as tf
except ImportError:
    import types

    tf = types.SimpleNamespace()
    tf.Tensor = None
    tf.RaggedTensor = None


def _is_composite_array(x):
    if isinstance(x, tf.RaggedTensor):
        return True
    if ivy.is_ivy_sparse_array(x) or ivy.is_native_sparse_array(x):
        return True
    return False


def _flatten_composite_array(x):
    if isinstance(x, tf.RaggedTensor):
        new_struc = [x.flat_values]
        for row_split in x.nested_row_splits:
            new_struc.append(row_split)
        return new_struc
    elif ivy.is_ivy_sparse_array(x) or ivy.is_native_sparse_array(x):
        return ivy.native_sparse_array_to_indices_values_and_shape(x)


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
