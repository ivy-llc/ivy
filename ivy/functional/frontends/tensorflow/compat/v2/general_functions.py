from ivy.functional.frontends.tensorflow.general_functions import convert_to_tensor as tf_convert_to_tensor, identity as tf_identity
from ivy.functional.frontends.tensorflow.linalg import tensordot as tf_tensordot, matmul as tf_matmul
from ivy.functional.frontends.tensorflow.math import tanh as tf_tanh, reduce_mean as tf_reduce_mean
from ivy.functional.frontends.tensorflow.dtypes import cast as tf_cast


convert_to_tensor = tf_convert_to_tensor
tensordot = tf_tensordot
tanh = tf_tanh
cast = tf_cast
identity = tf_identity
matmul = tf_matmul
reduce_mean = tf_reduce_mean
