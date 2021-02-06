"""
Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

relu = _tf.nn.relu
leaky_relu = _tf.nn.leaky_relu
tanh = _tf.nn.tanh
sigmoid = _tf.nn.sigmoid
softmax = _tf.nn.softmax
softplus = _tf.nn.softplus
