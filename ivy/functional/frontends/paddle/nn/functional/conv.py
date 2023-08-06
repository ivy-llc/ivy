# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ....torch.nn.functional import convolution_functions


def _channel_first_input(x, data_format):
    ndims = len(x.shape)
    dims = ndims - 2
    if 1 >= dims >= 3:
        raise ivy.utils.exceptions.IvyError(f"invalid for input with {dims} dims")
    # channel first input
    if data_format not in ["NCL", "NCHW", "NCDHW"]:
        if data_format in ["NLC", "NHWC", "NDHWC"]:
            x = ivy.permute_dims(x, axes=(0, ndims - 1, *range(1, ndims - 1)))
        else:
            raise ivy.utils.exceptions.IvyError(
                "data_format should be " + "'NCL' or 'NLC' "
                if dims == 1
                else (
                    "'NCHW' or 'NHWC' "
                    if dims == 2
                    else "'NCDHW' or 'NDHWC' " + f"but got data_format '{data_format}'"
                )
            )
    return x


def _conv(
    x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, data_format="NLC"
):
    x = _channel_first_input(x, data_format)
    return convolution_functions._conv(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
    data_format="NLC",
):
    x = _channel_first_input(x, data_format)
    return convolution_functions._conv_transpose(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv1d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NCL",
    name=None,
):
    return _conv(x, weight, bias, stride, padding, dilation, groups, data_format)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv2d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NCHW",
    name=None,
):
    return _conv(x, weight, bias, stride, padding, dilation, groups, data_format)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv3d(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    data_format="NCDHW",
    name=None,
):
    return _conv(x, weight, bias, stride, padding, dilation, groups, data_format)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv1d_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    output_size=None,
    data_format="NCL",
    name=None,
):
    return _conv_transpose(
        x, weight, bias, stride, padding, output_padding, dilation, groups, data_format
    )


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv2d_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
    output_size=None,
    data_format="NCHW",
    name=None,
):
    return _conv_transpose(
        x, weight, bias, stride, padding, output_padding, dilation, groups, data_format
    )


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def conv3d_transpose(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    output_size=None,
    data_format="NCDHW",
    name=None,
):
    return _conv_transpose(
        x, weight, bias, stride, padding, output_padding, dilation, groups, data_format
    )
