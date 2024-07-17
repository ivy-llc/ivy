import tensorflow


def tensorflow_is_native_array(x, /, *, exclusive=False):
    if isinstance(x, (tensorflow.Tensor, tensorflow.Variable, tensorflow.TensorArray)):
        if exclusive and isinstance(x, tensorflow.Variable):
            return False
        return True
    return False
