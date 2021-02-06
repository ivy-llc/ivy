"""
Collection of TensorFlow network layers, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

conv1d = lambda x, filters, strides, padding, data_format='NWC', dilations=1, _=None, _1=None: _tf.nn.conv1d(x, filters, strides, padding, data_format, dilations)
conv1d_transpose = lambda x, filters, strides, padding, output_shape, data_format='NWC', dilations=1, _=None, _1=None: _tf.nn.conv1d_transpose(x, filters, output_shape, strides, padding, data_format, dilations)
conv2d = lambda x, filters, strides, padding, data_format='NHWC', dilations=1, _=None, _1=None: _tf.nn.conv2d(x, filters, strides, padding, data_format, dilations)
conv2d_transpose = lambda x, filters, strides, padding, output_shape, data_format='NHWC', dilations=1, _=None, _1=None: _tf.nn.conv2d_transpose(x, filters, output_shape, strides, padding, data_format, dilations)


def depthwise_conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1, _=None, _1=None, _2=None):
    filters = _tf.expand_dims(filters, -1)
    strides = [1, strides, strides, 1]
    dilations = [dilations, dilations]
    return _tf.nn.depthwise_conv2d(x, filters, strides, padding, data_format, dilations)


# noinspection PyDefaultArgument
def conv3d(x, filters, strides, padding, data_format='NDHWC', dilations=1, _=None, _1=None):
    strides = [1]*2 + ([strides]*3 if isinstance(strides, int) else strides)
    dilations = [1]*2 + ([dilations]*3 if isinstance(dilations, int) else dilations)
    return _tf.nn.conv3d(x, filters, strides, padding, data_format, dilations)


conv3d_transpose = lambda x, filters, strides, padding, output_shape, data_format='NHWC', dilations=1, _=None, _1=None: _tf.nn.conv3d_transpose(x, filters, output_shape, strides, padding, data_format, dilations)
linear = lambda x, weight, bias, _=None: _tf.matmul(x, _tf.transpose(weight)) + bias
