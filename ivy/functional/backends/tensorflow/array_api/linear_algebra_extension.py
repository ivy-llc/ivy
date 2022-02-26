#global
import tensorflow as tf
from typing import Optional

def det(x:tf.Tensor,name:Optional[str]=None) \
    -> tf.Tensor:
    return tf.linalg.det(x,name)
