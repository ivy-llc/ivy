#global
import tensorflow as tf

def det(x:tf.Tensor) \
    -> tf.Tensor:
    return tf.linalg.det(x)
