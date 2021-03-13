"""
Collection of Ivy activation functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


def conv1d(x, filters, strides, padding, data_format='NWC', dilations=1, f=None):
    """
    Computes a 1-D convolution given 3-D input x and filters arrays.

    :param x: Input image *[batch_size,w,d_in]*.
    :type x: array
    :param filters: Convolution filters *[fw,d_in,d_out]*.
    :type filters: array
    :param strides: The stride of the sliding window for each dimension of input.
    :type strides: int or sequence of ints
    :param padding: "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension paddings.
    :type padding: string or sequence of ints
    :param data_format: "NWC" or "NCW". Defaults to "NWC".
    :type data_format: string
    :param dilations: The dilation factor for each dimension of input.
    :type dilations: int or sequence of ints
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The result of the convolution operation.
    """
    return _get_framework(x, f=f).conv1d(x, filters, strides, padding, data_format, dilations)


def conv1d_transpose(x, filters, strides, padding, output_shape=None, data_format='NWC', dilations=1, f=None):
    """
    Computes a 1-D transpose convolution given 3-D input x and filters arrays.

    :param x: Input image *[batch_size,w,d_in]*.
    :type x: array
    :param filters: Convolution filters *[fw,d_in,d_out]*.
    :type filters: array
    :param strides: The stride of the sliding window for each dimension of input.
    :type strides: int or sequence of ints
    :param padding: "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension paddings.
    :type padding: string or sequence of ints
    :param output_shape: Shape of the output
    :type output_shape: sequence of ints, needed for TensorFlow
    :param data_format: "NWC" or "NCW". Defaults to "NWC".
    :type data_format: string
    :param dilations: The dilation factor for each dimension of input.
    :type dilations: int or sequence of ints
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The result of the transpose convolution operation.
    """
    return _get_framework(x, f=f).conv1d_transpose(x, filters, strides, padding, output_shape, data_format, dilations)


def conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1, f=None):
    """
    Computes a 2-D convolution given 4-D input x and filters arrays.

    :param x: Input image *[batch_size,h,w,d_in]*.
    :type x: array
    :param filters: Convolution filters *[fh,fw,d_in,d_out]*.
    :type filters: array
    :param strides: The stride of the sliding window for each dimension of input.
    :type strides: int or sequence of ints
    :param padding: "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension paddings.
    :type padding: string or sequence of ints
    :param data_format: "NHWC" or "NCHW". Defaults to "NHWC".
    :type data_format: string
    :param dilations: The dilation factor for each dimension of input.
    :type dilations: int or sequence of ints
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The result of the convolution operation.
    """
    return _get_framework(x, f=f).conv2d(x, filters, strides, padding, data_format, dilations)


def conv2d_transpose(x, filters, strides, padding, output_shape=None, data_format='NHWC', dilations=1, f=None):
    """
    Computes a 2-D transpose convolution given 4-D input x and filters arrays.

    :param x: Input image *[batch_size,h,w,d_in]*.
    :type x: array
    :param filters: Convolution filters *[fh,fw,d_in,d_out]*.
    :type filters: array
    :param strides: The stride of the sliding window for each dimension of input.
    :type strides: int or sequence of ints
    :param padding: "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension paddings.
    :type padding: string or sequence of ints
    :param output_shape: Shape of the output
    :type output_shape: sequence of ints, needed for TensorFlow
    :param data_format: "NHWC" or "NCHW". Defaults to "NHWC".
    :type data_format: string
    :param dilations: The dilation factor for each dimension of input.
    :type dilations: int or sequence of ints
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The result of the transpose convolution operation.
    """
    return _get_framework(x, f=f).conv2d_transpose(x, filters, strides, padding, output_shape, data_format, dilations)


def depthwise_conv2d(x, filters, strides, padding, data_format='NHWC', dilations=1, f=None):
    """
    Computes a 2-D depthwise convolution given 4-D input x and filters arrays.

    :param x: Input image *[batch_size,h,w,d]*.
    :type x: array
    :param filters: Convolution filters *[fh,fw,d]*.
    :type filters: array
    :param strides: The stride of the sliding window for each dimension of input.
    :type strides: int or sequence of ints
    :param padding: "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension paddings.
    :type padding: string or sequence of ints
    :param data_format: "NHWC" or "NCHW". Defaults to "NHWC".
    :type data_format: string
    :param dilations: The dilation factor for each dimension of input.
    :type dilations: int or sequence of ints
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The result of the convolution operation.
    """
    return _get_framework(x, f=f).depthwise_conv2d(x, filters, strides, padding, data_format, dilations)


# noinspection PyDefaultArgument
def conv3d(x, filters, strides, padding, data_format='NDHWC', dilations=1, f=None):
    """
    Computes a 3-D convolution given 5-D input x and filters arrays.

    :param x: Input volume *[batch_size,d,h,w,d_in]*.
    :type x: array
    :param filters: Convolution filters *[fd,fh,fw,d_in,d_out]*.
    :type filters: array
    :param strides: The stride of the sliding window for each dimension of input.
    :type strides: sequence of ints
    :param padding: "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension paddings.
    :type padding: string or sequence of ints
    :param data_format: "NDHWC" or "NCDHW". Defaults to "NDHWC".
    :type data_format: string
    :param dilations: The dilation factor for each dimension of input.
    :type dilations: int or sequence of ints
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The result of the convolution operation.
    """
    return _get_framework(x, f=f).conv3d(x, filters, strides, padding, data_format, dilations)


def conv3d_transpose(x, filters, strides, padding, output_shape=None, data_format='NDHWC', dilations=1, f=None):
    """
    Computes a 3-D transpose convolution given 5-D input x and filters arrays.

    :param x: Input image *[batch_size,d,h,w,d_in]*.
    :type x: array
    :param filters: Convolution filters *[fd,fh,fw,d_in,d_out]*.
    :type filters: array
    :param strides: The stride of the sliding window for each dimension of input.
    :type strides: int or sequence of ints
    :param padding: "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension paddings.
    :type padding: string or sequence of ints
    :param output_shape: Shape of the output
    :type output_shape: sequence of ints, needed for TensorFlow
    :param data_format: "NDHWC" or "NCDHW". Defaults to "NDHWC".
    :type data_format: string
    :param dilations: The dilation factor for each dimension of input.
    :type dilations: int or sequence of ints
    :param f: Machine learning library. Inferred from Inputs if None.
    :type f: ml_framework, optional
    :return: The result of the transpose convolution operation.
    """
    return _get_framework(x, f=f).conv3d_transpose(x, filters, strides, padding, output_shape, data_format, dilations)


def linear(x, weight, bias, f=None):
    """
    Applies a linear transformation to the incoming data: y = x * t(weight) + bias,
    where t(...) indicates transpose.

    :param x: The input x compute linear transformation on. *[N,*,in_features]*
    :type x: array
    :param weight: The weight matrix. *[out_features,in_features]*
    :type weight: array
    :param bias: The bias vector. *[out_features]*
    :type bias: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Result array of the linear transformation. *[N,∗,out_features]*
    """
    return _get_framework(x, f=f).linear(x, weight, bias)
