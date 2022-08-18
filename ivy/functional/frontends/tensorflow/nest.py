# global
import ivy
import tensorflow as tf


RAGGED_TYPES = (tf.RaggedTensor,)
SPARSE_TYPES = (tf.SparseTensor,)


def flatten(structure, expand_composites=False):
    if expand_composites and isinstance(structure, RAGGED_TYPES):
        new_struc = []
        all_splits = [0]
        splits = []
        child_splits = []
        buffer = []
        for child in structure:
            splits.append(child.shape[0])
            if isinstance(child, RAGGED_TYPES):
                result = flatten(child, True)
                new_struc += ivy.to_list(result[0])
                child_splits.append(ivy.to_list(result[1]))
                buffer += result[2:]
            else:
                new_struc += ivy.to_list(child)
        if child_splits != []:
            combined_splits = child_splits[0]
            for s in child_splits[1:]:
                current = combined_splits[-1]
                for idx in s[1:]:
                    combined_splits.append(current + idx)
            buffer = [ivy.native_array(combined_splits, dtype="int64")] + buffer
        for idx in splits:
            all_splits.append(all_splits[-1] + idx)
        return [
            ivy.native_array(new_struc),
            ivy.native_array(all_splits, dtype="int64"),
        ] + buffer
    if expand_composites and isinstance(structure, SPARSE_TYPES):
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
