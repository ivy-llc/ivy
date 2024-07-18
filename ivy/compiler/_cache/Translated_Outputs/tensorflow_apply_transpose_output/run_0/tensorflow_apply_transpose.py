from .tensorflow__helpers import tensorflow_permute_dims


def tensorflow_apply_transpose(input, transpose, pt_to_tf=True):
    from .tensorflow_TransposeType import tensorflow_TransposeType

    if transpose is tensorflow_TransposeType.NO_TRANSPOSE:
        return input
    if transpose is tensorflow_TransposeType.CONV1D:
        axes = (0, 2, 1) if pt_to_tf else (0, 2, 1)
    elif transpose is tensorflow_TransposeType.CONV2D:
        axes = (0, 2, 3, 1) if pt_to_tf else (0, 3, 1, 2)
    elif transpose is tensorflow_TransposeType.CONV3D:
        axes = (0, 2, 3, 4, 1) if pt_to_tf else (0, 4, 1, 2, 3)
    input = tensorflow_permute_dims(input, axes=axes)
    return input
