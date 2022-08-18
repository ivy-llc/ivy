# global
import ivy
import tensorflow as tf


RAGGED_TYPES = (tf.RaggedTensor,)
# SPARSE_TYPES = (tf.)


def flatten(structure, expand_composites=False):
    if expand_composites and isinstance(structure, RAGGED_TYPES):
        new_struc = []
        all_splits = [0]
        splits = []
        buffer = []
        for child in structure:
            splits.append(child.shape[0])
            if isinstance(child, RAGGED_TYPES):
                result = flatten(child, True)
                new_struc += ivy.to_list(result[0])
                buffer += result[1:]
            else:
                new_struc += ivy.to_list(child)
        for idx in splits:
            all_splits.append(all_splits[-1] + idx)
        return [ivy.native_array(new_struc), ivy.native_array(all_splits)] + buffer
    # if expand_composites and isinstance(structure, tf.RaggedTensor):
    #     print('yes!')
    #     return structure
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
