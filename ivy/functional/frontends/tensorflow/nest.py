# local
import ivy.functional.frontends.tensorflow.ragged as ragged_tf
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
import ivy


# try:
#     import tensorflow as tf
# except ImportError:
#     import types
#
#     tf = types.SimpleNamespace()
#     tf.Tensor = None
#     tf.RaggedTensor = None


def _is_composite_array(x):
    if isinstance(x, ragged_tf.RaggedTensor):
        return True
    if ivy.is_ivy_sparse_array(x) or ivy.is_native_sparse_array(x):
        return True
    return False


def _flatten_composite_array(x, expand_composites=False):
    if isinstance(x, ragged_tf.RaggedTensor):
        if not expand_composites:
            return x
        new_struc = [x.flat_values]
        for row_split in x.nested_row_splits:
            new_struc.append(row_split)
        return new_struc
    elif ivy.is_ivy_sparse_array(x) or ivy.is_native_sparse_array(x):
        return ivy.native_sparse_array_to_indices_values_and_shape(x)


@to_ivy_arrays_and_back
def flatten(structure, expand_composites=False):
    if expand_composites and _is_composite_array(structure):
        return _flatten_composite_array(structure, expand_composites=expand_composites)
    elif isinstance(structure, (tuple, list)):
        return [x for child in structure for x in flatten(child)]
    elif isinstance(structure, dict):
        return [x for key in sorted(structure) for x in flatten(structure[key])]
    return [structure]
