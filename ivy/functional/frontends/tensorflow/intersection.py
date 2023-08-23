import ivy.functional.backends.tensorflow as ivy_tf #import tensorflow backend

def intersection(x,y): #implementing the intersection function
    return ivy_tf.minimum(x,y)