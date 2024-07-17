import tensorflow


def tensorflow_get_num_dims(x, /, *, as_array=False):
    return (
        tensorflow.cast(tensorflow.shape(tensorflow.shape(x))[0], tensorflow.int64)
        if as_array
        else int(tensorflow.shape(tensorflow.shape(x)))
    )
