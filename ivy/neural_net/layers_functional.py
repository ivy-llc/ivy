"""
Collection of Ivy neural network layers in functional form.
"""

# local
import ivy
from ivy.framework_handler import get_framework as _get_framework


# Linear #
# -------#

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


# Convolutions #
# -------------#

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


# LSTM #
# -----#

def lstm_update(x, init_h, init_c, kernel, recurrent_kernel, bias=None, recurrent_bias=None):
    """
    Perform long-short term memory update by unrolling time dimension of input array.

    :param x: input tensor of LSTM layer *[batch_shape, t, in]*.
    :type x: array
    :param init_h: initial state tensor for the cell output *[batch_shape, out]*.
    :type init_h: array
    :param init_c: initial state tensor for the cell hidden state *[batch_shape, out]*.
    :type init_c: array
    :param kernel: weights for cell kernel *[in, 4 x out]*.
    :type kernel: array
    :param recurrent_kernel: weights for cell recurrent kernel *[out, 4 x out]*.
    :type recurrent_kernel: array
    :param bias: bias for cell kernel *[4 x out]*.
    :type bias: array
    :param recurrent_bias: bias for cell recurrent kernel *[4 x out]*.
    :type recurrent_bias: array
    :return: hidden state for all timesteps *[batch_shape,t,out]* and cell state for last timestep *[batch_shape,out]*
    """

    # get shapes
    x_shape = list(x.shape)
    batch_shape = x_shape[:-2]
    timesteps = x_shape[-2]
    input_channels = x_shape[-1]
    x_flat = ivy.reshape(x, (-1, input_channels))

    # input kernel
    Wi = kernel
    Wi_x = ivy.reshape(ivy.matmul(x_flat, Wi) + (bias if bias is not None else 0),
                        batch_shape + [timesteps, -1])
    Wii_x, Wif_x, Wig_x, Wio_x = ivy.split(Wi_x, 4, -1)

    # recurrent kernel
    Wh = recurrent_kernel

    # lstm states
    ht = init_h
    ct = init_c

    # lstm outputs
    ot = x
    hts_list = list()

    # unrolled time dimension with lstm steps
    for Wii_xt, Wif_xt, Wig_xt, Wio_xt in zip(ivy.unstack(Wii_x, axis=-2), ivy.unstack(Wif_x, axis=-2),
                                              ivy.unstack(Wig_x, axis=-2), ivy.unstack(Wio_x, axis=-2)):
        htm1 = ht
        ctm1 = ct

        Wh_htm1 = ivy.matmul(htm1, Wh) + (recurrent_bias if recurrent_bias is not None else 0)
        Whi_htm1, Whf_htm1, Whg_htm1, Who_htm1 = ivy.split(Wh_htm1, num_sections=4, axis=-1)

        it = ivy.sigmoid(Wii_xt + Whi_htm1)
        ft = ivy.sigmoid(Wif_xt + Whf_htm1)
        gt = ivy.tanh(Wig_xt + Whg_htm1)
        ot = ivy.sigmoid(Wio_xt + Who_htm1)
        ct = ft * ctm1 + it * gt
        ht = ot * ivy.tanh(ct)

        hts_list.append(ivy.expand_dims(ht, -2))

    return ivy.concatenate(hts_list, -2), ct
