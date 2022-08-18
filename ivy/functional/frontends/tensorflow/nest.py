# global
import ivy
import tensorflow as tf


RAGGED_TYPES = (tf.RaggedTensor,)
# SPARSE_TYPES = (tf.)


def flatten(structure, expand_composites=False):
    if expand_composites and isinstance(structure, RAGGED_TYPES):
        new_struc = []
        for child in structure:
            result = flatten(child, True)
            new_struc.append(result[0])
        structure = new_struc
        new_struc = []
        for child in structure:
            new_struc += ivy.to_list(child)
        return [ivy.native_array(new_struc)]
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
