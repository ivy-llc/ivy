import tensorflow


def tensorflow__reshape_fortran_tf(x, shape):
    if len(x.shape) > 0:
        x = tensorflow.transpose(x)
    return tensorflow.transpose(tensorflow.reshape(x, shape[::-1]))
