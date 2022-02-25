import tensorflow as tf
from tensorflow.python.types.core import Tensor

def builtin_lt(self: Tensor,other: Tensor)\
        -> Tensor :
    if hasattr(self,'dtype') and hasattr(other,'dtype'):
        promoted_type = tf.experimental.numpy.promote_types(self.dtype,other.dtype)
        self = tf.cast(self,promoted_type)
        other = tf.cast(other,promoted_type)
    return self.__lt__(other)