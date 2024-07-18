import tensorflow


def tensorflow_is_variable(x, /, *, exclusive=False):
    return isinstance(x, tensorflow.Variable)
