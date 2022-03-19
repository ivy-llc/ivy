"""
Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as tf

relu = tf.nn.relu
leaky_relu = tf.nn.leaky_relu
gelu = lambda x, approximate=True: tf.nn.gelu(x, approximate)
tanh = tf.nn.tanh
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax
softplus = tf.nn.softplus
